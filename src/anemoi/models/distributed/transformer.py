# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.utils import get_memory_format

LOGGER = logging.getLogger(__name__)


def _headsalltoall(input_: Tensor, shapes: list, group: Optional[ProcessGroup] = None) -> Tensor:
    """Apply all_to_all along the head dimension.

    Split input along dimension dim_split and join after all_to_all along dimesion
    dim_concatenate.
    """
    comm_size = dist.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if comm_size == 1:
        return input_

    # get input format
    input_format = get_memory_format(input_)

    input_list = [x.contiguous() for x in torch.tensor_split(input_, comm_size, dim=-3)]  # do we need contiguous?

    input_shape = [x.shape for x in input_list]  # (b ... h n c)
    heads_per_rank = [x.shape[-3] for x in input_list]
    channels_per_rank = [x.shape[-1] for x in input_list]
    seq_per_rank = [x[0] for x in shapes]

    output_list = [
        torch.empty(
            (*input_shape[rank][:-3], heads_per_rank[rank], seq_per_rank[rank], channels_per_rank[rank]),
            dtype=input_.dtype,
            layout=input_.layout,
            device=input_.device,
            memory_format=input_format,
        )
        for rank in range(comm_size)
    ]

    dist.all_to_all(output_list, input_list, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    return torch.cat(output_list, dim=-2).contiguous(memory_format=input_format)


def _seqalltoall(input_: Tensor, shapes: list, group: Optional[ProcessGroup] = None) -> Tensor:
    """Apply all_to_all along the sequence dimension.

    Split input along dimension dim_split and join after all_to_all along dimesion
    dim_concatenate.
    """
    comm_size = dist.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if comm_size == 1:
        return input_

    comm_rank = dist.get_rank(group=group)

    # get input format
    input_format = get_memory_format(input_)

    input_list = [x.contiguous() for x in torch.tensor_split(input_, comm_size, dim=-2)]  # do we need contiguous?

    output_list = [torch.empty_like(input_list[comm_rank]) for _ in range(comm_size)]

    dist.all_to_all(output_list, input_list, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    return torch.cat(output_list, dim=-3).contiguous(memory_format=input_format)


def _halo_exchange(input_: Tensor, halo_size: int, mgroup: ProcessGroup, bwd: bool = False) -> Tensor:
    """Exchange halo regions between ranks.

    Send halo regions to the left and right and receive from the right and left and extend the input tensor.
    Expected format is (batch_size, sequence_length, channels).

    Parameters
    ----------
    input_ : Tensor
        Input tensor
    halo_size : int
        Halo size (left, right)
    mgroup : ProcessGroup
        Model communication group
    bwd : bool
        Flag to indicate if backward pass

    Returns
    -------
    Tensor
        Extended input tensor
    """
    end = input_.shape[-2]

    left_halo = input_[:, :halo_size, :]
    right_halo = input_[:, end - halo_size :, :]

    left_send = input_[:, halo_size : 2 * halo_size, :]
    right_send = input_[:, end - 2 * halo_size : end - halo_size, :]

    if bwd:  # reverse halo exchange
        left_halo, left_send = left_send, left_halo
        right_halo, right_send = right_send, right_halo

    my_rank = dist.get_rank(mgroup)
    group_size = dist.get_world_size(mgroup)
    left_rank = dist.get_rank(mgroup) - 1
    right_rank = dist.get_rank(mgroup) + 1

    if my_rank % 2 != 0:
        # send left (can't be rank 0)
        dist.send(left_send, left_rank, group=mgroup)
        # receive left
        dist.recv(left_halo, left_rank, group=mgroup)

        if my_rank != group_size - 1:
            # send right
            dist.send(right_send, right_rank, group=mgroup)
            # receive right
            dist.recv(right_halo, right_rank, group=mgroup)
    else:
        if my_rank != group_size - 1:
            # receive right
            dist.recv(right_halo, right_rank, group=mgroup)
            # send right
            dist.send(right_send, right_rank, group=mgroup)
        if my_rank != 0:
            # receive left
            dist.recv(left_halo, left_rank, group=mgroup)
            # send left
            dist.send(left_send, left_rank, group=mgroup)

    return input_


def shard_heads(input_: Tensor, shapes: list, mgroup: ProcessGroup) -> Tensor:
    """Sync tensor.

    Gathers e.g query, key or value tensor along sequence dimension via all to all communication
    and shards along head dimension for parallel self-attention computation.
    Expected format is (batch_size, ... heads, sequence_length, channels)

    Parameters
    ----------
    input_ : Tensor
        Input
    shapes: list
        shapes of shards
    mgroup : ProcessGroup
        model communication group

    Returns
    -------
    Tensor
        Sharded heads.
    """
    return _SplitHeadsParallelSection.apply(input_, shapes, mgroup)


def shard_sequence(input_: Tensor, shapes: list, mgroup: ProcessGroup) -> Tensor:
    """Sync tensor.

    Gathers e.g query, key or value tensor along head dimension via all to all communication
    and shards along sequence dimension for parallel mlp and layernorm computation.
    Expected format is (batch_size, ... heads, sequence_length, channels)

    Parameters
    ----------
    input_ : Tensor
        Input
    shapes: list
        shapes of shards
    mgroup : ProcessGroup
        model communication group

    Returns
    -------
    Tensor
        Sharded sequence
    """
    return _SplitSequenceParallelSection.apply(input_, shapes, mgroup)


def halo_exchange(x: Tensor, halo_size: int, mgroup: ProcessGroup) -> None:
    """Exchange halo regions between ranks,

    Parameters
    ----------
    x : Tensor
        Input tensor
    halo_size : int
        Halo size (left, right)
    mgroup : ProcessGroup
        Model communication group
    """
    # pad tensor with halo regions
    halo_size_left = halo_size if (mgroup and dist.get_rank(mgroup) != 0) else 0
    halo_size_right = halo_size if (mgroup and dist.get_rank(mgroup) != dist.get_world_size(mgroup) - 1) else 0
    x_pad = torch.nn.functional.pad(x, pad=(0, 0, halo_size_left, halo_size_right), mode="constant", value=0)

    out = _HaloExchange.apply(x_pad, halo_size, mgroup)

    return out, halo_size_left, halo_size_right


class _SplitHeadsParallelSection(torch.autograd.Function):
    """Sync the input from parallel section."""

    @staticmethod
    def forward(ctx, input_, shapes_, mgroup_):
        ctx.shapes = shapes_
        ctx.comm_group = mgroup_
        if mgroup_:
            return _headsalltoall(input_, shapes_, group=mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.comm_group:
            return (
                _seqalltoall(grad_output, ctx.shapes, group=ctx.comm_group),
                None,
                None,
            )
        return grad_output, None, None


class _SplitSequenceParallelSection(torch.autograd.Function):
    """Sync the input from parallel section."""

    @staticmethod
    def forward(ctx, input_, shapes_, mgroup_):
        ctx.shapes = shapes_
        ctx.comm_group = mgroup_
        if mgroup_:
            return _seqalltoall(input_, shapes_, group=mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.comm_group:
            return (
                _headsalltoall(grad_output, ctx.shapes, group=ctx.comm_group),
                None,
                None,
            )
        return grad_output, None, None


class _HaloExchange(torch.autograd.Function):
    """Exchange halo regions between ranks."""

    @staticmethod
    def forward(ctx, input_, halo_size_, mgroup_):
        ctx.halo_size = halo_size_
        ctx.mgroup = mgroup_

        if mgroup_:
            return _halo_exchange(input_, halo_size_, mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.mgroup:
            return (
                # not sure if this works yet, need to test
                _halo_exchange(grad_output, ctx.halo_size, ctx.mgroup, bwd=True),
                None,
                None,
            )

        return grad_output, None, None
