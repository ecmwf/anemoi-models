# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup


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


def shard_tensor(
    input_: Tensor, dim: int, shapes: tuple, mgroup: ProcessGroup, gather_in_backward: bool = True
) -> Tensor:
    """Shard tensor.

    Keeps only part of the tensor that is relevant for the current rank.

    Parameters
    ----------
    input_ : Tensor
        Input
    dim : int
        dimension along which to shard
    shapes : tuple
        Shapes of sharded Tensors
    mgroup : ProcessGroup
        model communication group
    gather_in_backward : bool
        perform gather in backward, default True

    Returns
    -------
    Tensor
        Sharded tensor.
    """
    return _ShardParallelSection.apply(input_, dim, shapes, gather_in_backward, mgroup)


def gather_tensor(input_: Tensor, dim: int, shapes: tuple, mgroup: ProcessGroup) -> Tensor:
    """Gather tensor.

    Gathers tensor shards from ranks.

    Parameters
    ----------
    input_ : Tensor
        Input
    dim : int
        dimension along which to gather
    shapes : tuple
        Shapes of sharded Tensors
    mgroup : ProcessGroup
        model communication group

    Returns
    -------
    Tensor
        Gathered tensor.
    """
    return _GatherParallelSection.apply(input_, dim, shapes, mgroup)


def reduce_tensor(input_: Tensor, mgroup: ProcessGroup) -> Tensor:
    """Reduce tensor.

    Reduces tensor across ranks.

    Parameters
    ----------
    input_ : Tensor
        Input
    mgroup : ProcessGroup
        model communication group

    Returns
    -------
    Tensor
        Reduced tensor.
    """
    return _ReduceParallelSection.apply(input_, mgroup)


def sync_tensor(input_: Tensor, dim: int, shapes: tuple, mgroup: ProcessGroup) -> Tensor:
    """Sync tensor.

    Perform a gather in the forward pass and an allreduce followed by a split in the backward pass.

    Parameters
    ----------
    input_ : Tensor
        Input
    dim : int
        dimension along which to gather
    shapes : tuple
        Shapes of sharded Tensors
    mgroup : ProcessGroup
        model communication group

    Returns
    -------
    Tensor
        Synced tensor.
    """
    return _SyncParallelSection.apply(input_, dim, shapes, mgroup)


def reduce_shard_tensor(input_: Tensor, dim: int, shapes: tuple, mgroup: ProcessGroup) -> Tensor:
    """Reduces and then shards tensor.

    Perform an allreduce followed by a split in the forward pass and a gather in the backward pass.

    Parameters
    ----------
    input_ : Tensor
        Input
    dim : int
        dimension along which to gather
    shapes : tuple
        Shapes of sharded Tensors
    mgroup : ProcessGroup
        model communication group

    Returns
    -------
    Tensor
        Reduced sharded tensor.
    """
    return _ReduceShardParallelSection.apply(input_, dim, shapes, mgroup)


def get_shape_shards(tensor: Tensor, dim: int, mgroup: Optional[ProcessGroup] = None) -> list:
    """Get shape of tensor shards."""
    assert dim < tensor.dim(), f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}"

    comm_size = 1 if not mgroup else dist.get_world_size(group=mgroup)
    return [list(x.shape) for x in torch.tensor_split(tensor, comm_size, dim=dim)]


def change_channels_in_shape(shape_list: list, channels: int) -> list:
    """Change the number of channels in the tensor shape definition list."""
    return [x[:-1] + [channels] for x in shape_list] if shape_list else []


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


class _SyncParallelSection(torch.autograd.Function):
    """Sync the input from parallel section."""

    @staticmethod
    def forward(ctx, input_, dim_, shapes_, mgroup_):
        ctx.dim = dim_
        ctx.comm_group = mgroup_
        ctx.shapes = shapes_
        if mgroup_:
            return _gather(input_, dim_, shapes_, group=mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.comm_group:
            grad_output = _reduce(grad_output, group=ctx.comm_group)
            return (
                _split(grad_output, ctx.dim, ctx.shapes, group=ctx.comm_group),
                None,
                None,
                None,
            )
        return grad_output, None, None, None


class _ReduceShardParallelSection(torch.autograd.Function):
    """All-reduce and shard the input from the parallel section."""

    @staticmethod
    def forward(ctx, input_, dim_, shapes_, mgroup_):
        ctx.dim = dim_
        ctx.comm_group = mgroup_
        ctx.shapes = shapes_
        if mgroup_:
            input_ = _reduce(input_, group=mgroup_)
            return _split(input_, dim_, shapes_, group=mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.comm_group:
            return (
                _gather(grad_output, ctx.dim, ctx.shapes, group=ctx.comm_group),
                None,
                None,
                None,
            )
        return grad_output, None, None, None


class _ShardParallelSection(torch.autograd.Function):
    """Split the input and keep only the relevant chunck to the rank."""

    @staticmethod
    def forward(ctx, input_, dim_, shapes_, gather_in_backward_, mgroup_):
        ctx.dim = dim_
        ctx.comm_group = mgroup_
        ctx.shapes = shapes_
        ctx.gather_in_backward = gather_in_backward_
        if mgroup_:
            return _split(input_, dim_, shapes_, group=mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.comm_group:
            return (
                _gather(
                    grad_output, ctx.dim, ctx.shapes, gather_in_backward=ctx.gather_in_backward, group=ctx.comm_group
                ),
                None,
                None,
                None,
                None,
            )
        return grad_output, None, None, None, None


class _GatherParallelSection(torch.autograd.Function):
    """Gather the input from parallel section and concatenate."""

    @staticmethod
    def forward(ctx, input_, dim_, shapes_, mgroup_):
        ctx.dim = dim_
        ctx.comm_group = mgroup_
        ctx.shapes = shapes_
        if mgroup_:
            return _gather(input_, dim_, shapes_, group=mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.comm_group:
            return (
                _split(grad_output, ctx.dim, ctx.shapes, group=ctx.comm_group),
                None,
                None,
                None,
            )
        return grad_output, None, None, None


class _ReduceParallelSection(torch.autograd.Function):
    """All-reduce the input from the parallel section."""

    @staticmethod
    def forward(ctx, input_, mgroup_):
        if mgroup_:
            return _reduce(input_, group=mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


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


def _split(input_: Tensor, dim_: int, shapes_: tuple, group: Optional[ProcessGroup] = None) -> Tensor:
    """Split the tensor along dim and keep the relevant slice."""
    # get input format
    input_format = get_memory_format(input_)

    # Bypass the function if we are using only 1 GPU.
    comm_size = dist.get_world_size(group=group)
    if comm_size == 1:
        return input_

    # sanity checks
    assert dim_ < input_.dim(), f"Error, cannot split along {dim_} for tensor with {input_.dim()} dimensions."

    input_list = torch.split(input_, [x[dim_] for x in shapes_], dim=dim_)

    rank = dist.get_rank(group=group)
    output = input_list[rank].contiguous(memory_format=input_format)

    return output


def _gather(
    input_: Tensor,
    dim_: int,
    shapes: tuple,
    gather_in_backward: Optional[bool] = True,
    group: Optional[ProcessGroup] = None,
) -> Tensor:
    """Gather tensors and concatenate along the last dimension."""
    # get input format
    input_format = get_memory_format(input_)

    comm_size = dist.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if comm_size == 1:
        return input_

    # sanity checks
    assert dim_ < input_.dim(), f"Error, cannot gather along {dim_} for tensor with {input_.dim()} dimensions."

    # Size and dimension.
    comm_rank = dist.get_rank(group=group)

    input_ = input_.contiguous(memory_format=input_format)
    tensor_list = [
        torch.empty(
            shapes[rank], dtype=input_.dtype, layout=input_.layout, device=input_.device, memory_format=input_format
        )
        for rank in range(comm_size)
    ]

    tensor_list[comm_rank] = input_
    if gather_in_backward:
        dist.all_gather(tensor_list, input_, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=dim_).contiguous(memory_format=input_format)

    return output


def _reduce(input_: Tensor, use_fp32: Optional[bool] = True, group: Optional[ProcessGroup] = None) -> Tensor:
    """All-reduce the input tensor across model parallel group."""
    comm_size = dist.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if comm_size == 1:
        return input_

    # All-reduce.
    if use_fp32:
        dtype = input_.dtype
        inputf_ = input_.float()
        dist.all_reduce(inputf_, group=group)
        input_ = inputf_.to(dtype)
    else:
        dist.all_reduce(input_, group=group)

    return input_


def get_memory_format(tensor: Tensor):
    """Helper routine to get the memory format."""
    if tensor.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    return torch.contiguous_format
