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
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import get_shape_shards

LOGGER = logging.getLogger(__name__)


class AnemoiModelMultEncProcDec(nn.Module):
    """Message passing graph neural network.
    
    This class receives a dictionary of tensors and returns a dictionary of tensors.

    This class supports multiple encoders.

    This class supports multiple decoders.

    """

    def __init__(
        self,
        *,
        config: DotDict,
        data_indices: dict,
        graph_data: HeteroData,
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        config : DotDict
            Job configuration
        data_indices : dict
            Data indices
        graph_data : HeteroData
            Graph definition
        """
        super().__init__()

        self._graph_data = graph_data
        self._graph_name_input_data = [config.graph.data]
        self._graph_name_output_data = [config.graph.data]
        self._graph_name_hidden = config.graph.hidden
        self._all_graph_data = list(set(self._graph_name_input_data) | set(self._graph_name_output_data))
        self._all_graph_data.append(self._graph_name_hidden)

        self._calculate_shapes_and_indices(data_indices)
        self._assert_matching_indices(data_indices)

        self.multi_step = config.training.multistep_input

        # Register lat/lon of nodes
        for nodes_name in self._all_graph_data:
            self._register_latlon(nodes_name, nodes_name)

        num_nodes = {name: self._graph_data[name].num_nodes for name in self._graph_mesh_names}
        num_node_features = {name: 2 * self._graph_data[name]["x"].shape[1] for name in self._graph_mesh_names}

        self.num_channels = config.model.num_channels

        input_dim = self.multi_step * self.num_input_channels

        # Encoder data -> hidden
        self.encoders = nn.ModuleDict()
        for input_name in self._input_model_data:
            self.encoders[input_name] = instantiate(
                config.model.encoder,
                in_channels_src=input_dim + num_node_features[input_name],
                in_channels_dst=num_node_features[self._graph_name_hidden],
                hidden_dim=self.num_channels,
                sub_graph=self._graph_data[(self._graph_name_data, "to", self._graph_name_hidden)],
                src_grid_size=num_nodes[input_name],
                dst_grid_size=num_nodes[self._graph_name_hidden],
            )

        # Processor hidden -> hidden
        self.processor = instantiate(
            config.model.processor,
            num_channels=self.num_channels,
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_hidden)],
            src_grid_size=self._hidden_grid_size,
            dst_grid_size=self._hidden_grid_size,
        )

        # Decoder hidden -> data
        self.decoders = nn.ModuleDict()
        for output_name in self._output_model_data:
            self.decoders[output_name] = instantiate(
                config.model.decoder,
                in_channels_src=self.num_channels,
                in_channels_dst=input_dim + num_node_features[output_name],
                hidden_dim=self.num_channels,
                out_channels_dst=self.num_output_channels,
                sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_data)],
                src_grid_size=num_nodes[self._graph_name_hidden],
                dst_grid_size=num_nodes[output_name],
            )

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        self.num_input_channels = len(data_indices.internal_model.input)
        self.num_output_channels = len(data_indices.internal_model.output)
        self._internal_input_idx = data_indices.internal_model.input.prognostic
        self._internal_output_idx = data_indices.internal_model.output.prognostic

    def _assert_matching_indices(self, data_indices: dict) -> None:

        assert len(self._internal_output_idx) == len(data_indices.internal_model.output.full) - len(
            data_indices.internal_model.output.diagnostic
        ), (
            f"Mismatch between the internal data indices ({len(self._internal_output_idx)}) and "
            f"the internal output indices excluding diagnostic variables "
            f"({len(data_indices.internal_model.output.full) - len(data_indices.internal_model.output.diagnostic)})",
        )
        assert len(self._internal_input_idx) == len(
            self._internal_output_idx,
        ), f"Internal model indices must match {self._internal_input_idx} != {self._internal_output_idx}"

    def _register_latlon(self, name: str, nodes: str) -> None:
        """Register lat/lon buffers.

        Parameters
        ----------
        name : str
            Name to store the lat-lon coordinates of the nodes.
        nodes : str
            Name of nodes to map
        """
        coords = self._graph_data[nodes].x
        sin_cos_coords = torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
        self.register_buffer(f"latlons_{name}", sin_cos_coords, persistent=True)

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

    def forward(self, x: dict[str, Tensor], model_comm_group: Optional[ProcessGroup] = None) -> dict[str, Tensor]:
        batch_size = x[list(x.keys())[0]].shape[0]
        ensemble_size = x[list(x.keys())[0]].shape[2]

        # add data positional info (lat/lon)
        x_data_latent = {}
        for in_data in self._graph_name_input_data:
            x_data_latent[in_data] = torch.cat(
                (
                    einops.rearrange(x[in_data], "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                    getattr(self, f"trainable_{in_data}")(getattr(self, f"latlons_{in_data}"), batch_size=batch_size),
                ),
                dim=-1,  # feature dimension
            )

        x_hidden_latent = self.trainable_hidden(self.latlons_hidden, batch_size=batch_size)

        # get shard shapes
        shard_shapes_data = {}
        for in_data in self._graph_input_meshes:
            shard_shapes_data[in_data] = get_shape_shards(x_data_latent[in_data], 0, model_comm_group)
        shard_shapes_hidden = get_shape_shards(x_hidden_latent, 0, model_comm_group)

        # Run encoder
        x_latents = []
        for in_data, encoder in self.encoders.items():
            x_data_latent[in_data], x_latent = self._run_mapper(
                encoder,
                (x_data_latent[in_data], x_hidden_latent),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_data[in_data], shard_shapes_hidden),
                model_comm_group=model_comm_group,
            )
            x_latents.append(x_latent)

        # TODO: This operation can be a desing choice (sum, mean, attention, ...)
        x_latent = torch.stack(x_latents).sum(dim=0) if len(x_latents) > 1 else x_latents[0]

        x_latent_proc = self.processor(
            x_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        # add skip connection (hidden -> hidden)
        x_latent_proc = x_latent_proc + x_latent

        # Run decoder
        x_out = {}
        for out_data, decoder in self.decoders.items():
            x_out[out_data] = self._run_mapper(
                decoder,
                (x_latent_proc, x_data_latent[out_data]),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hidden, shard_shapes_data[out_data]),
                model_comm_group=model_comm_group,
            )

            x_out[out_data] = (
                einops.rearrange(
                    x_out[out_data],
                    "(batch ensemble grid) vars -> batch ensemble grid vars",
                    batch=batch_size,
                    ensemble=ensemble_size,
                )
                .to(dtype=x.dtype)
                .clone()
            )

            # residual connection (just for the prognostic variables)
            if out_data in self._graph_name_input_data:
                x_out[out_data][..., self._internal_output_idx] += x[:, -1, :, :, self._internal_input_idx]

        return x_out
