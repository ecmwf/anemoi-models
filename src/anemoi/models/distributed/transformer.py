# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
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

    left_halo_slice = slice(0, halo_size)
    right_halo_slice = slice(end - halo_size, end)
    left_send_slice = slice(halo_size, 2 * halo_size)
    right_send_slice = slice(end - 2 * halo_size, end - halo_size)

    if bwd:  # reverse halo exchange direction
        left_halo_slice, left_send_slice = left_send_slice, left_halo_slice
        right_halo_slice, right_send_slice = right_send_slice, right_halo_slice

    left_send = input_[:, left_send_slice, :]
    right_send = input_[:, right_send_slice, :]
    left_halo = torch.empty_like(right_send, device=input_.device)
    right_halo = torch.empty_like(left_send, device=input_.device)

    global_rank = dist.get_rank()
    local_rank = dist.get_rank(mgroup)
    group_size = dist.get_world_size(mgroup)
    left_rank = global_rank - 1 if local_rank > 0 else None
    right_rank = global_rank + 1 if local_rank < group_size - 1 else None

    match os.environ.get("HALO_COMM", "SENDRECV"):
        case "SENDRECV":
            if local_rank % 2 != 0:
                if left_rank is not None:
                    dist.send(left_send, left_rank, group=mgroup)
                    dist.recv(left_halo, left_rank, group=mgroup)
                if right_rank is not None:
                    dist.send(right_send, right_rank, group=mgroup)
                    dist.recv(right_halo, right_rank, group=mgroup)
            else:
                if right_rank is not None:
                    dist.recv(right_halo, right_rank, group=mgroup)
                    dist.send(right_send, right_rank, group=mgroup)
                if left_rank is not None:
                    dist.recv(left_halo, left_rank, group=mgroup)
                    dist.send(left_send, left_rank, group=mgroup)
        case "ISENDRECV":
            reqs = []
            if local_rank % 2 != 0:
                if left_rank is not None:
                    reqs.append(dist.isend(left_send, left_rank, group=mgroup))
                    reqs.append(dist.irecv(left_halo, left_rank, group=mgroup))
                if right_rank is not None:
                    reqs.append(dist.isend(right_send, right_rank, group=mgroup))
                    reqs.append(dist.irecv(right_halo, right_rank, group=mgroup))
            else:
                if right_rank is not None:
                    reqs.append(dist.irecv(right_halo, right_rank, group=mgroup))
                    reqs.append(dist.isend(right_send, right_rank, group=mgroup))
                if left_rank is not None:
                    reqs.append(dist.irecv(left_halo, left_rank, group=mgroup))
                    reqs.append(dist.isend(left_send, left_rank, group=mgroup))
            for req in reqs:
                req.wait()
        case "ALLGATHER":
            combined_send = torch.cat([left_send, right_send], dim=1).contiguous()
            halos = [torch.empty_like(combined_send) for _ in range(group_size)]
            dist.all_gather(halos, combined_send, group=mgroup)
            left_halo = halos[local_rank - 1][:, halo_size:, :] if local_rank > 0 else None
            right_halo = halos[local_rank + 1][:, :halo_size, :] if local_rank < group_size - 1 else None
        case "ALLTOALL":
            input_list = [torch.empty(1, device=input_.device) for _ in range(group_size)]
            if left_rank is not None:
                input_list[left_rank] = left_send
            if right_rank is not None:
                input_list[right_rank] = right_send
            output_list = [torch.empty_like(input_i, device=input_.device) for input_i in input_list]
            dist.all_to_all(output_list, input_list, group=mgroup)

            if left_rank is not None:
                left_halo = output_list[left_rank]
            if right_rank is not None:
                right_halo = output_list[right_rank]
        case _:
            raise ValueError(f"Unknown halo communication strategy {os.environ['HALO_COMM']}")

    if bwd:
        # remove gradient contribution from send regions and add halo regions
        if left_rank is not None:
            input_[:, left_send_slice, :] = 0
            input_[:, left_halo_slice, :] += left_halo
        if right_rank is not None:
            input_[:, right_send_slice, :] = 0
            input_[:, right_halo_slice, :] += right_halo
    else:
        if left_rank is not None:
            input_[:, left_halo_slice, :] = left_halo
        if right_rank is not None:
            input_[:, right_halo_slice, :] = right_halo

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
