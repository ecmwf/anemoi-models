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
import torch
from anemoi.utils.config import DotConfig
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.distributed.shapes import get_shape_shards
from anemoi.models.layers.graph import TrainableTensor

LOGGER = logging.getLogger(__name__)


class AnemoiModelEncProcDec(nn.Module):
    """Message passing graph neural network."""

    def __init__(
        self,
        *,
        config: DotConfig,
        data_indices: IndexCollection,
        graph_data: dict,
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        data_indices : IndexCollection
            Data indices
        graph_data : dict
            Graph definition
        """
        super().__init__()

        self._graph_name_hidden = config.graphs.hidden_mesh.name
        self._graph_mesh_names = [name for name in graph_data if isinstance(name, str)]
        self._graph_input_meshes = [
            k[0] for k in graph_data if isinstance(k, tuple) and k[2] == self._graph_name_hidden and k[2] != k[0]
        ]
        self._graph_output_meshes = [
            k[2] for k in graph_data if isinstance(k, tuple) and k[0] == self._graph_name_hidden and k[2] != k[0]
        ]

        self._calculate_shapes_and_indices(data_indices)
        self._assert_matching_indices(data_indices)

        self.multi_step = config.training.multistep_input

        self._define_tensor_sizes(config, graph_data)

        self._create_trainable_attributes()

        # Register lat/lon
        for mesh_key in self._graph_mesh_names:
            self._register_latlon(mesh_key, graph_data[mesh_key]["coords"])

        self.num_channels = config.model.num_channels

        input_dim = self.multi_step * self.num_input_channels

        # Encoder data -> hidden
        self.encoders = nn.ModuleDict()
        for in_mesh in self._graph_input_meshes:
            self.encoders[in_mesh] = instantiate(
                config.model.encoder,
                in_channels_src=input_dim + self.num_node_features[in_mesh] + self.num_trainable_params[in_mesh],
                in_channels_dst=self.num_node_features[self._graph_name_hidden]
                + self.num_trainable_params[self._graph_name_hidden],
                hidden_dim=self.num_channels,
                sub_graph=graph_data[(in_mesh, "to", self._graph_name_hidden)],
                src_grid_size=self.num_nodes[in_mesh],
                dst_grid_size=self.num_nodes[self._graph_name_hidden],
            )

        # Processor hidden -> hidden
        self.processor = instantiate(
            config.model.processor,
            num_channels=self.num_channels,
            sub_graph=graph_data.get((self._graph_name_hidden, "to", self._graph_name_hidden), None),
            src_grid_size=self.num_nodes[self._graph_name_hidden],
            dst_grid_size=self.num_nodes[self._graph_name_hidden],
        )

        # Decoder hidden -> data
        self.decoders = nn.ModuleDict()
        for out_mesh in self._graph_output_meshes:
            self.decoders[out_mesh] = instantiate(
                config.model.decoder,
                in_channels_src=self.num_channels,
                in_channels_dst=input_dim + self.num_node_features[out_mesh] + self.num_trainable_params[out_mesh],
                hidden_dim=self.num_channels,
                out_channels_dst=self.num_output_channels,
                sub_graph=graph_data[(self._graph_name_hidden, "to", out_mesh)],
                src_grid_size=self.num_nodes[self._graph_name_hidden],
                dst_grid_size=self.num_nodes[out_mesh],
            )

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        self.num_input_channels = len(data_indices.model.input)
        self.num_output_channels = len(data_indices.model.output)
        self._internal_input_idx = data_indices.model.input.prognostic
        self._internal_output_idx = data_indices.model.output.prognostic

    def _assert_matching_indices(self, data_indices: dict) -> None:

        assert len(self._internal_output_idx) == len(data_indices.model.output.full) - len(
            data_indices.model.output.diagnostic
        ), (
            f"Mismatch between the internal data indices ({len(self._internal_output_idx)}) and the output indices excluding "
            f"diagnostic variables ({len(data_indices.model.output.full) - len(data_indices.model.output.diagnostic)})",
        )
        assert len(self._internal_input_idx) == len(
            self._internal_output_idx,
        ), f"Model indices must match {self._internal_input_idx} != {self._internal_output_idx}"

    def _define_tensor_sizes(self, config: DotConfig, graph_data: dict) -> None:
        # Define Sizes of different tensors
        self.num_nodes = {name: graph_data[name]["coords"].shape[0] for name in self._graph_mesh_names}
        self.num_node_features = {name: 2 * graph_data[name]["coords"].shape[1] for name in self._graph_mesh_names}
        self.num_trainable_params = {
            name: config.model.trainable_parameters["data" if name != "hidden" else name]
            for name in self._graph_mesh_names
        }

    def _create_trainable_attributes(self) -> None:
        """Create all trainable attributes."""
        self.trainable_tensors = nn.ModuleDict()
        for mesh in self._graph_mesh_names:
            self.trainable_tensors[mesh] = TrainableTensor(
                trainable_size=self.num_trainable_params[mesh], tensor_size=self.num_nodes[mesh]
            )

    def _register_latlon(self, name: str, coords: torch.Tensor) -> None:
        """Register lat/lon buffers.

        Parameters
        ----------
        name : str
            Name of grid to map
        coords: torch.Tensor
            Coordinates of the grid
        """
        self.register_buffer(
            f"latlons_{name}", torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1), persistent=True
        )

    def _run_mapper(
        self,
        mapper: nn.Module,
        data: tuple[Tensor],
        batch_size: int,
        shard_shapes: tuple[tuple[int, int], tuple[int, int]],
        model_comm_group: Optional[ProcessGroup] = None,
        use_reentrant: bool = False,
    ) -> Tensor:
        """Run mapper with activation checkpoint.

        Parameters
        ----------
        mapper : nn.Module
            Which processor to use
        data : tuple[Tensor]
            tuple of data to pass in
        batch_size: int,
            Batch size
        shard_shapes : tuple[tuple[int, int], tuple[int, int]]
            Shard shapes for the data
        model_comm_group : ProcessGroup
            model communication group, specifies which GPUs work together
            in one model instance
        use_reentrant : bool, optional
            Use reentrant, by default False

        Returns
        -------
        Tensor
            Mapped data
        """
        return checkpoint(
            mapper,
            data,
            batch_size=batch_size,
            shard_shapes=shard_shapes,
            model_comm_group=model_comm_group,
            use_reentrant=use_reentrant,
        )

    def forward(self, x: Tensor, model_comm_group: Optional[ProcessGroup] = None) -> Tensor:
        batch_size = x.shape[0]
        ensemble_size = x.shape[2]

        # add data positional info (lat/lon)
        x_data_latent = {}
        for in_mesh in self._graph_input_meshes:
            x_data_latent[in_mesh] = torch.cat(
                (
                    einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                    self.trainable_tensors[in_mesh](getattr(self, f"latlons_{in_mesh}"), batch_size=batch_size),
                ),
                dim=-1,  # feature dimension
            )

        x_hidden_latent = self.trainable_tensors[self._graph_name_hidden](self.latlons_hidden, batch_size=batch_size)

        # get shard shapes
        shard_shapes_data = {name: get_shape_shards(data, 0, model_comm_group) for name, data in x_data_latent.items()}
        shard_shapes_hidden = get_shape_shards(x_hidden_latent, 0, model_comm_group)

        # Run encoders
        x_latents = []
        for in_data_name, encoder in self.encoders.items():
            x_data_latent[in_data_name], x_latent = self._run_mapper(
                encoder,
                (x_data_latent[in_data_name], x_hidden_latent),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_data[in_data_name], shard_shapes_hidden),
                model_comm_group=model_comm_group,
            )
            x_latents.append(x_latent)

        # TODO: This operation can be a design choice (sum, mean, attention, ...)
        x_latent = torch.stack(x_latents).sum(dim=0) if len(x_latents) > 1 else x_latents[0]

        x_latent_proc = self.processor(
            x_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        # add skip connection (hidden -> hidden)
        x_latent_proc = x_latent_proc + x_latent

        # Run decoders
        x_out = {}
        for out_data_name, decoder in self.decoders.items():
            x_out[out_data_name] = self._run_mapper(
                decoder,
                (x_latent_proc, x_data_latent[out_data_name]),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hidden, shard_shapes_data[out_data_name]),
                model_comm_group=model_comm_group,
            )

            x_out[out_data_name] = (
                einops.rearrange(
                    x_out[out_data_name],
                    "(batch ensemble grid) vars -> batch ensemble grid vars",
                    batch=batch_size,
                    ensemble=ensemble_size,
                )
                .to(dtype=x.dtype)
                .clone()
            )

            if out_data_name in self._graph_input_meshes:  # check if the mesh is in the input meshes
                # residual connection (just for the prognostic variables)
                x_out[out_data_name][..., self._internal_output_idx] += x[:, -1, :, :, self._internal_input_idx]

        return x_out[self._graph_output_meshes[0]]
