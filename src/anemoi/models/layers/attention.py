# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging
import math
from typing import Optional

import einops
import torch
from packaging import version
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.transformer import shard_heads
from anemoi.models.distributed.transformer import shard_sequence

LOGGER = logging.getLogger(__name__)


class MultiHeadSelfAttention(nn.Module):
    """Multi Head Self Attention Pytorch Layer using flash attention, see https://github.com/Dao-AILab/flash-attention"""

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        bias: bool = False,
        is_causal: bool = False,
        window_size: Optional[int] = None,
        dropout_p: float = 0.0,
        attention_implementation: str = "flex attention",
        softcap: float = None,
        use_alibi_slopes: bool = None,
    ):
        """Initialize MultiHeadSelfAttention.

        Parameters
        ----------
        num_heads : int
            number of heads
        embed_dim : int
            embedding dimension
        bias : bool, optional
            bias, by default False
        is_causal : bool, optional
            apply causal attention mask, by default False
        window_size : Optional[int], optional
            window_size, by default None
        dropout_p : float, optional
            dropout probability, by default 0.0
        softcap : float, optional
            Anything > 0 activates softcapping attention, by default None
        attention_implementation: str, optional
            A predefined string which selects which underlying attention
            implementation, by default "flex attention"
        use_alibi_slopes : bool, optional
            Adds bias of (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            to the attention score of query i and key j, where alibi_slope
            is calculated using get_alibi_slopes, by default None
        """
        super().__init__()

        assert (
            embed_dim % num_heads == 0
        ), f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})"

        self.attention_implementation = attention_implementation
        self.use_alibi_slopes = use_alibi_slopes
        self.set_attention_function()

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads  # q k v
        self.window_size = window_size
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.softcap = softcap

        if self.use_alibi_slopes is not None:
            self.alibi_slopes = get_alibi_slopes(num_heads)
            assert self.alibi_slopes.shape[0] == num_heads
        else:
            self.alibi_slopes = None

        self.lin_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)

        self.projection = nn.Linear(embed_dim, embed_dim, bias=True)

    def set_attention_function(self):
        attn_funcs = {
            "flash attention": FlashAttentionWrapper,
            "flex attention": FlexAttentionWrapper,
            "scaled dot product attention": TorchAttentionWrapper,
        }
        if self.attention_implementation in attn_funcs:
            LOGGER.info(f"attention.py: using {self.attention_implementation}")
            # initalise the attn func here
            self.attention = attn_funcs[self.attention_implementation]()
        else:
            # Requested attn implementation is not supported
            raise SystemExit(
                f"attention.py: Error! {self.attention_implementation} not supported. \
                    please change model.processor.attention_implementation in the config to one of: {attn_funcs.keys()}"
            )

    def forward(
        self, x: Tensor, shapes: list, batch_size: int, model_comm_group: Optional[ProcessGroup] = None
    ) -> Tensor:

        query, key, value = self.lin_qkv(x).chunk(3, -1)

        if model_comm_group:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), "Only batch size of 1 is supported when model is sharded accross GPUs"

        query, key, value = (
            einops.rearrange(
                t,
                "(batch grid) (heads vars) -> batch heads grid vars",
                batch=batch_size,
                heads=self.num_heads,
            )
            for t in (query, key, value)
        )

        query = shard_heads(query, shapes=shapes, mgroup=model_comm_group)
        key = shard_heads(key, shapes=shapes, mgroup=model_comm_group)
        value = shard_heads(value, shapes=shapes, mgroup=model_comm_group)
        dropout_p = self.dropout_p if self.training else 0.0

        out = self.attention(
            query,
            key,
            value,
            batch_size,
            causal=False,
            window_size=self.window_size,
            dropout_p=dropout_p,
            softcap=self.softcap,
            alibi_slopes=self.alibi_slopes,
        )

        out = shard_sequence(out, shapes=shapes, mgroup=model_comm_group)
        out = einops.rearrange(out, "batch heads grid vars -> (batch grid) (heads vars)")

        out = self.projection(out)

        return out


class TorchAttentionWrapper(nn.Module):
    """Wrapper for Pytorch dot product attention"""

    def __init__(self):
        super().__init__()

        from torch.nn.functional import scaled_dot_product_attention

        self.attention = scaled_dot_product_attention

    def forward(
        self,
        query,
        key,
        value,
        batch_size: int,
        causal=False,
        window_size=None,
        dropout_p=0.0,
        softcap=None,
        alibi_slopes=None,
    ):
        if softcap is not None:
            SystemError(
                "Error. Softcap not supported by Pytorchs SDPA. please switch to flash attention or disable softcap."
            )
        if alibi_slopes is not None:
            SystemError(
                "Error. Alibi slopes not supported by Pytorchs SDPA. please switch to flash attention or disable alibi slopes."
            )
        if window_size is not None:
            SystemError(
                "Error. Sliding window not supported by Pytorchs SDPA. please switch to flash attention or disable sliding window."
            )

        return self.attention(
            query,
            key,
            value,
            is_causal=causal,
            dropout_p=dropout_p,
        )


class FlexAttentionWrapper(nn.Module):
    """Wrapper for Pytorch Flex attention."""

    def __init__(self):
        super().__init__()

        if version.parse(torch.__version__) < version.parse("2.5.0"):
            raise SystemExit("Error: torch version is too low. Update to 2.5.0 or higher to use Flex Attention.")

        # we compile flex attn once at the first iteration
        # This is bc we need to know the seq len to compute the mask mod for sliding window
        self.is_attn_compiled = False

    def forward(
        self,
        query,
        key,
        value,
        batch_size: int,
        causal: bool = False,
        window_size: int = None,
        dropout_p: float = 0.0,
        softcap: float = None,
        alibi_slopes: torch.Tensor = None,
    ):

        if alibi_slopes is not None:
            SystemExit("Error. Alibi_slopes not yet implemented in FlexAttn in Anemoi.")
        if softcap is not None:
            SystemExit("Error. Softcap not yet implemented in FlexAttn in Anemoi.")
        if dropout_p != 0.0:
            SystemExit("Error. Dropout not yet implemented in FlexAttn in Anemoi.")
        if causal:
            SystemExit("Error. Causal not yet implemented in FlexAttn in Anemoi.")

        # This assumes seq_len never changes
        # across iterations and stages
        # could add something like
        #   if query.shape[2] != prev_seq_len:
        #       self.is_attn_compiled = False
        # To trigger a recompilation
        if not self.is_attn_compiled:
            import functools

            from torch.nn.attention.flex_attention import create_block_mask  # should this be after the version check?
            from torch.nn.attention.flex_attention import flex_attention

            def sliding_window_mask(b, h, q_idx, kv_idx):
                return abs(q_idx - kv_idx) <= window_size

            seq_len = query.shape[2]
            self.block_mask = create_block_mask(
                sliding_window_mask, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=True
            )
            self.attention = functools.partial(
                flex_attention, block_mask=self.block_mask
            )  # Cache the block mask (recomended in attn blog post)
            self.attention = torch.compile(self.attention)
            self.is_attn_compiled = True

        # TODO test how this impacts scaling at large model counts
        torch._dynamo.config.optimize_ddp = False
        out = self.attention(query, key, value)
        torch._dynamo.config.optimize_ddp = True

        return out


class FlashAttentionWrapper(nn.Module):
    """Wrapper for Flash attention."""

    def __init__(self):
        super().__init__()
        import flash_attn

        if version.parse(flash_attn.__version__) < version.parse("2.6.0"):
            raise SystemExit("Error: Flash-attn version is too low. Update to 2.6.0 or higher.")
        else:
            self.attention = flash_attn.flash_attn_func

    def forward(
        self,
        query,
        key,
        value,
        batch_size: int,
        causal: bool = False,
        window_size: int = None,
        dropout_p: float = 0.0,
        softcap: float = None,
        alibi_slopes: torch.Tensor = None,
    ):
        query, key, value = (
            einops.rearrange(t, "batch heads grid vars -> batch grid heads vars") for t in (query, key, value)
        )

        alibi_slopes = alibi_slopes.repeat(batch_size, 1).to(query.device) if self.use_alibi_slopes else None

        out = self.attention(
            query,
            key,
            value,
            causal=False,
            window_size=(window_size, window_size),
            dropout_p=dropout_p,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
        )
        out = einops.rearrange(out, "batch grid heads vars -> batch heads grid vars")
        return out


def get_alibi_slopes(num_heads: int) -> Tensor:
    """Calculates linearly decreasing slopes for alibi attention.

    Parameters
    ----------
    num_heads : int
        number of attention heads

    Returns
    -------
    Tensor
        aLiBi slopes
    """
    n = 2 ** math.floor(math.log2(num_heads))
    slope_0 = 2.0 ** (-8.0 / n)
    alibi_slopes = torch.pow(slope_0, torch.arange(1, 1 + n))
    if n < num_heads:
        slope_hat_0 = 2.0 ** (-4.0 / n)
        alibi_slopes_hat = torch.pow(slope_hat_0, torch.arange(1, 1 + 2 * (num_heads - n), 2))
        alibi_slopes = torch.cat([alibi_slopes, alibi_slopes_hat])
    return alibi_slopes
