# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import einops
import torch
from torch import Tensor
from torch import nn
from torch_geometric.data import HeteroData


class TrainableTensor(nn.Module):
    """Trainable Tensor Module."""

    def __init__(self, tensor_size: int, trainable_size: int) -> None:
        """Initialize TrainableTensor."""
        super().__init__()

        if trainable_size > 0:
            trainable = nn.Parameter(
                torch.empty(
                    tensor_size,
                    trainable_size,
                ),
            )
            nn.init.constant_(trainable, 0)
        else:
            trainable = None
        self.register_parameter("trainable", trainable)

    def forward(self, x: Tensor, batch_size: int) -> Tensor:
        latent = [einops.repeat(x, "e f -> (repeat e) f", repeat=batch_size)]
        if self.trainable is not None:
            latent.append(einops.repeat(self.trainable.to(x.device), "e f -> (repeat e) f", repeat=batch_size))
        return torch.cat(
            latent,
            dim=-1,  # feature dimension
        )


class NamedNodesAttributes(torch.nn.Module):
    """Named Node Attributes Module."""

    def __init__(self, num_trainable_params: int, graph_data: HeteroData) -> None:
        """Initialize NamedNodesAttributes."""
        super().__init__()

        self.num_trainable_params = num_trainable_params
        self.register_fixed_attributes(graph_data)

        self.trainable_tensors = nn.ModuleDict()
        for nodes_name in self.nodes_names:
            self.register_coordinates(nodes_name, graph_data[nodes_name].x)
            self.register_tensor(nodes_name)

    def register_fixed_attributes(self, graph_data: HeteroData) -> None:
        """Register fixed attributes."""
        self.nodes_names = list(graph_data.node_types)
        self.num_nodes = {nodes_name: graph_data[nodes_name].num_nodes for nodes_name in self.nodes_names}
        self.coord_dims = {nodes_name: 2 * graph_data[nodes_name].x.shape[1] for nodes_name in self.nodes_names}
        self.attr_ndims = {
            nodes_name: self.coord_dims[nodes_name] + self.num_trainable_params for nodes_name in self.nodes_names
        }

    def register_coordinates(self, name: str, node_coords: torch.Tensor) -> None:
        """Register coordinates."""
        sin_cos_coords = torch.cat([torch.sin(node_coords), torch.cos(node_coords)], dim=-1)
        self.register_buffer(f"latlons_{name}", sin_cos_coords, persistent=True)

    def register_tensor(self, name: str) -> None:
        """Register a trainable tensor."""
        self.trainable_tensors[name] = TrainableTensor(self.num_nodes[name], self.num_trainable_params)

    def forward(self, name: str, batch_size: int) -> Tensor:
        """Forward pass."""
        latlons = getattr(self, f"latlons_{name}")
        return self.trainable_tensors[name](latlons, batch_size)
