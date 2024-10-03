# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

import logging
import math
from typing import Optional

import einops
import torch
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
        use_flash_attention: bool = False,
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
        use_alibi_slopes : bool, optional
            Adds bias of (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            to the attention score of query i and key j, where alibi_slope
            is calculated using get_alibi_slopes, by default None
        """
        super().__init__()

        assert (
            embed_dim % num_heads == 0
        ), f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})"

        self.use_flash_attention = use_flash_attention
        self.set_attention_function()

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads  # q k v
        self.window_size = (window_size, window_size)  # flash attention
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.softcap = softcap
        self.use_alibi_slopes = use_alibi_slopes

        if self.use_alibi_slopes is not None:
            self.alibi_slopes = get_alibi_slopes(num_heads)
            assert self.alibi_slopes.shape[0] == num_heads

        self.lin_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)

        self.projection = nn.Linear(embed_dim, embed_dim, bias=True)

    def set_attention_function(self):

        if self.use_flash_attention:
            from flash_attn import flash_attn_func

            self.attention = flash_attn_func
        else:
            from torch.nn.functional import scaled_dot_product_attention

            self.attention = scaled_dot_product_attention

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

        if self.use_flash_attention:
            query, key, value = (
                einops.rearrange(t, "batch heads grid vars -> batch grid heads vars") for t in (query, key, value)
            )

            alibi_slopes = self.alibi_slopes.repeat(batch_size, 1).to(query.device) if self.use_alibi_slopes else None

            out = self.attention(
                query,
                key,
                value,
                causal=False,
                window_size=self.window_size,
                dropout_p=dropout_p,
                softcap=self.softcap,
                alibi_slopes=alibi_slopes,
            )
            out = einops.rearrange(out, "batch grid heads vars -> batch heads grid vars")
        else:
            out = self.attention(
                query,
                key,
                value,
                is_causal=False,
                dropout_p=dropout_p,
            )

        out = shard_sequence(out, shapes=shapes, mgroup=model_comm_group)
        out = einops.rearrange(out, "batch heads grid vars -> (batch grid) (heads vars)")

        out = self.projection(out)

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
