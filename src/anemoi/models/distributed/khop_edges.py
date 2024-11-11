# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Optional
from typing import Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.typing import Adj
from torch_geometric.utils import bipartite_subgraph
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import mask_to_index


def get_k_hop_edges(nodes: Tensor, edge_attr: Tensor, edge_index: Adj, num_hops: int = 1) -> tuple[Adj, Tensor]:
    """Return 1 hop subgraph.

    Parameters
    ----------
    nodes : Tensor
        destination nodes
    edge_attr : Tensor
        edge attributes
    edge_index : Adj
        edge index
    num_hops: int, Optional, by default 1
        number of required hops

    Returns
    -------
    tuple[Adj, Tensor]
        K-hop subgraph of edge index and edge attributes
    """
    _, edge_index_k, _, edge_mask_k = k_hop_subgraph(
        node_idx=nodes, num_hops=num_hops, edge_index=edge_index, directed=True
    )

    return edge_attr[mask_to_index(edge_mask_k)], edge_index_k


def sort_edges_1hop_sharding(
    num_nodes: Union[int, tuple[int, int]],
    edge_attr: Tensor,
    edge_index: Adj,
    mgroup: Optional[ProcessGroup] = None,
) -> tuple[Adj, Tensor, list, list]:
    """Rearanges edges into 1 hop neighbourhoods for sharding across GPUs.

    Parameters
    ----------
    num_nodes : Union[int, tuple[int, int]]
        Number of (target) nodes in Graph
    edge_attr : Tensor
        edge attributes
    edge_index : Adj
        edge index
    mgroup : ProcessGroup
        model communication group

    Returns
    -------
    tuple[Adj, Tensor, list, list]
        edges sorted according to k hop neigh., edge attributes of sorted edges,
        shapes of edge indices for partitioning between GPUs, shapes of edge attr for partitioning between GPUs
    """
    if mgroup:
        num_chunks = dist.get_world_size(group=mgroup)

        edge_attr_list, edge_index_list = sort_edges_1hop_chunks(num_nodes, edge_attr, edge_index, num_chunks)

        edge_index_shapes = [x.shape for x in edge_index_list]
        edge_attr_shapes = [x.shape for x in edge_attr_list]

        return torch.cat(edge_attr_list, dim=0), torch.cat(edge_index_list, dim=1), edge_attr_shapes, edge_index_shapes

    return edge_attr, edge_index, [], []


def sort_edges_1hop_chunks(
    num_nodes: Union[int, tuple[int, int]], edge_attr: Tensor, edge_index: Adj, num_chunks: int
) -> tuple[list[Tensor], list[Adj]]:
    """Rearanges edges into 1 hop neighbourhood chunks.

    Parameters
    ----------
    num_nodes : Union[int, tuple[int, int]]
        Number of (target) nodes in Graph, tuple for bipartite graph
    edge_attr : Tensor
        edge attributes
    edge_index : Adj
        edge index
    num_chunks : int
        number of chunks used if mgroup is None

    Returns
    -------
    tuple[list[Tensor], list[Adj]]
        list of sorted edge attribute chunks, list of sorted edge_index chunks
    """
    if isinstance(num_nodes, int):
        node_chunks = torch.arange(num_nodes, device=edge_index.device).tensor_split(num_chunks)
    else:
        nodes_src = torch.arange(num_nodes[0], device=edge_index.device)
        node_chunks = torch.arange(num_nodes[1], device=edge_index.device).tensor_split(num_chunks)

    edge_index_list = []
    edge_attr_list = []
    for node_chunk in node_chunks:
        if isinstance(num_nodes, int):
            edge_attr_chunk, edge_index_chunk = get_k_hop_edges(node_chunk, edge_attr, edge_index)
        else:
            edge_index_chunk, edge_attr_chunk = bipartite_subgraph(
                (nodes_src, node_chunk),
                edge_index,
                edge_attr,
                size=(num_nodes[0], num_nodes[1]),
            )
        edge_index_list.append(edge_index_chunk)
        edge_attr_list.append(edge_attr_chunk)

    return edge_attr_list, edge_index_list
