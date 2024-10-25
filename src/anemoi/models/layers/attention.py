# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
from typing import Optional

import einops
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
except ImportError:
    _FLEX_ATTENTION_AVAILABLE = False
else:
    import functools
    from torch import abs, compile, _dynamo
    import os
    _FLEX_ATTENTION_AVAILABLE = True
    

try:
    from flash_attn import flash_attn_func as attn_func
except ImportError:
    from torch.nn.functional import scaled_dot_product_attention as attn_func

    _FLASH_ATTENTION_AVAILABLE = False
else:
    _FLASH_ATTENTION_AVAILABLE = True

from anemoi.models.distributed.transformer import shard_heads
from anemoi.models.distributed.transformer import shard_sequence
from anemoi.models.layers.utils import calculate_seq_len

LOGGER = logging.getLogger(__name__)


class MultiHeadSelfAttention(nn.Module):
    """Multi Head Self Attention Pytorch Layer."""

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        bias: bool = False,
        is_causal: bool = False,
        window_size: Optional[int] = None,
        dropout_p: float = 0.0,
        resolution: str = 'X0',
    ):
        super().__init__()

        assert (
            embed_dim % num_heads == 0
        ), f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})"

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads  # q k v
        self.window_size = (window_size, window_size)  # flash attention
        self.dropout_p = dropout_p
        self.is_causal = is_causal

        self.lin_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.attention = attn_func
        
        self.resolution = resolution
        
        self.use_document_masking = False
        self.is_attn_compiled = False
        self.compile_at_runtime=False
        self.use_flex_Attn=False

        if _FLEX_ATTENTION_AVAILABLE and (os.environ.get("FLEX_ATTN", "") != "" ):
            self.use_flex_Attn = True
            
        if self.use_flex_Attn:
            if not self.compile_at_runtime:
                LOGGER.info("Using Flex attn")
                #LOGGER.info(f"self.num_heads {self.num_heads} self.embed_dim {self.embed_dim} self.head_dim {self.head_dim} self.dropout {self.dropout_p}")
            
                if self.use_document_masking:
                    
                    def document_causal_mask(b, h, q_idx, kv_idx):
                        causal_mask = q_idx >= kv_idx
                        document_mask = document_id[q_idx] == document_id[kv_idx]
                        return causal_mask & document_mask

                
                elif window_size != None:
                    def sliding_window(b, h, q_idx, kv_idx):
                        return abs(q_idx - kv_idx) <= window_size

                    seq_len=calculate_seq_len(resolution=self.resolution)
                    LOGGER.debug(f"grid points = {seq_len} for {self.resolution} resolution")

                    # B and H can be None here because they are uniform, so the block mask can just be broadcast to these dims
                    #TODO check if B != 1, does it have to be set?
                    self.block_mask = create_block_mask(sliding_window, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len,_compile=True)
                    self.attention = functools.partial(flex_attention, block_mask=self.block_mask) #Cache the block mask (attn blog post)
                else:
                    self.attention = flex_attention
                self.attention = compile(self.attention) #Must be compiled, otherwise entire seq_len^2 aray is materilised in memory -> OOM
                self.is_attn_compiled=True

            if (self.is_causal):
                LOGGER.error("Causal not yet supported when using flex_attn (but this would be an easy add). Please rerun with 'is_causal = False'")
            if (self.dropout_p != 0.0):
                LOGGER.error("Dropout not yet supported when using flex_attn. Please rerun with 'dropout_p = 0.0'")

        if not _FLASH_ATTENTION_AVAILABLE:
            LOGGER.warning("Flash attention not available, falling back to pytorch scaled_dot_product_attention")

        self.projection = nn.Linear(embed_dim, embed_dim, bias=True)

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
        
        
        if (self.use_flex_Attn and not self.is_attn_compiled):
            LOGGER.info("Compiling Flex Attn at runtime")
            seq_len=x.shape[0]
            def sliding_window(b, h, q_idx, kv_idx):
                return abs(q_idx - kv_idx) <= self.window_size[0]
            self.block_mask = create_block_mask(sliding_window, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len,_compile=True)
            self.attention = functools.partial(flex_attention, block_mask=self.block_mask) #Cache the block mask (attn blog post)
            self.attention = compile(self.attention)
            self.is_attn_compiled = True
                
        query = shard_heads(query, shapes=shapes, mgroup=model_comm_group)
        key = shard_heads(key, shapes=shapes, mgroup=model_comm_group)
        value = shard_heads(value, shapes=shapes, mgroup=model_comm_group)
        dropout_p = self.dropout_p if self.training else 0.0

        if _FLEX_ATTENTION_AVAILABLE and (os.environ.get("FLEX_ATTN", "") != "" ):
            _dynamo.config.optimize_ddp = False
            out = self.attention(query, key, value)
            _dynamo.config.optimize_ddp = True
        
        elif _FLASH_ATTENTION_AVAILABLE:
            query, key, value = (
                einops.rearrange(t, "batch heads grid vars -> batch grid heads vars") for t in (query, key, value)
            )
            out = self.attention(query, key, value, causal=False, window_size=self.window_size, dropout_p=dropout_p)
            out = einops.rearrange(out, "batch grid heads vars -> batch heads grid vars")
        else:
            out = self.attention(
                query,
                key,
                value,
                is_causal=False,
                dropout_p=dropout_p,
            )  # expects (batch heads grid variable) format

        out = shard_sequence(out, shapes=shapes, mgroup=model_comm_group)
        out = einops.rearrange(out, "batch heads grid vars -> (batch grid) (heads vars)")

        out = self.projection(out)

        return out
