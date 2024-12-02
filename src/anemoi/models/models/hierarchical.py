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

from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec

LOGGER = logging.getLogger(__name__)


class AnemoiModelEncProcDecHierarchical(AnemoiModelEncProcDec):
    """Message passing hierarchical graph neural network."""

    graph_data: HeteroData
    _graph_name_data: str
    _graph_name_hidden: list[str]
    multi_step: int
    num_channels: dict[str, int]
    node_attributes: NamedNodesAttributes
    num_input_channels: int
    num_output_channels: int
    _internal_input_idx: list[int]
    _internal_output_idx: list[int]

    def set_graph_parameters(self, graph_data: HeteroData, model_config: DotDict) -> None:
        super().set_graph_parameters(graph_data, model_config)
        self.num_hidden = len(self._graph_name_hidden)

    def set_model_parameters(self, model_config: DotDict) -> None:
        self.level_process = model_config.model.enable_hierarchical_level_processing
        self.num_layers_processors = model_config.model.level_process_num_layers
        self.multi_step = model_config.training.multistep_input
        if isinstance(num_channels := model_config.model.num_channels, int):
            LOGGER.info("An increasing number of channels, doubling at each hierarchy level, is used.")
            self.num_channels = {  # dim. of features at each depth
                hidden: model_config.model.num_channels * (2**i) for i, hidden in enumerate(self._graph_hidden_names)
            }
        else:
            assert (
                len(num_channels) == self.num_hidden
            ), f"num_channels ({num_channels}) length must be equal to the number of hierarchy levels ({self.num_hidden})"
            self.num_channels = model_config.model.num_channels

    def instantiate_processor(self, model_config: DotDict) -> None:
        # Level processors
        self.down_level_processor = nn.ModuleDict()
        if self.level_process:
            self.up_level_processor = nn.ModuleDict()

            for i in range(0, self.num_hidden - 1):
                nodes_name = self._graph_name_hidden[i]

                self.down_level_processor[nodes_name] = instantiate(
                    model_config.model.processor,
                    num_channels=self.num_channels[nodes_name],
                    sub_graph=self._graph_data[(nodes_name, "to", nodes_name)],
                    src_grid_size=self.node_attributes.num_nodes[nodes_name],
                    dst_grid_size=self.node_attributes.num_nodes[nodes_name],
                    num_layers=self.num_layers_processors,
                )

                self.up_level_processor[nodes_name] = instantiate(
                    model_config.model.processor,
                    num_channels=self.num_channels[nodes_name],
                    sub_graph=self._graph_data[(nodes_name, "to", nodes_name)],
                    src_grid_size=self.node_attributes.num_nodes[nodes_name],
                    dst_grid_size=self.node_attributes.num_nodes[nodes_name],
                    num_layers=self.num_layers_processors,
                )

        # Downscale
        self.downscale = nn.ModuleDict()

        for i in range(0, self.num_hidden - 1):
            src_nodes_name = self._graph_name_hidden[i]
            dst_nodes_name = self._graph_name_hidden[i + 1]

            self.downscale[src_nodes_name] = instantiate(
                model_config.model.encoder,
                in_channels_src=self.num_channels[src_nodes_name],
                in_channels_dst=self.node_attributes.attr_ndims[dst_nodes_name],
                hidden_dim=self.num_channels[dst_nodes_name],
                sub_graph=self._graph_data[(src_nodes_name, "to", dst_nodes_name)],
                src_grid_size=self.node_attributes.num_nodes[src_nodes_name],
                dst_grid_size=self.node_attributes.num_nodes[dst_nodes_name],
            )

        # Process top level
        nodes_name = self._graph_name_hidden[self.num_hidden - 1]
        self.down_level_processor[nodes_name] = instantiate(
            model_config.model.processor,
            num_channels=self.num_channels[nodes_name],
            sub_graph=self._graph_data[(nodes_name, "to", nodes_name)],
            src_grid_size=self.node_attributes.num_nodes[nodes_name],
            dst_grid_size=self.node_attributes.num_nodes[nodes_name],
            num_layers=self.num_layers_processors,
        )

        # Upscale
        self.upscale = nn.ModuleDict()

        for i in range(1, self.num_hidden):
            src_nodes_name = self._graph_name_hidden[i]
            dst_nodes_name = self._graph_name_hidden[i - 1]

            self.upscale[src_nodes_name] = instantiate(
                model_config.model.decoder,
                in_channels_src=self.num_channels[src_nodes_name],
                in_channels_dst=self.num_channels[dst_nodes_name],
                hidden_dim=self.num_channels[src_nodes_name],
                out_channels_dst=self.num_channels[dst_nodes_name],
                sub_graph=self._graph_data[(src_nodes_name, "to", dst_nodes_name)],
                src_grid_size=self.node_attributes.num_nodes[src_nodes_name],
                dst_grid_size=self.node_attributes.num_nodes[dst_nodes_name],
            )

    def process(
        self,
        x: dict[str, Tensor],
        batch_size: int,
        shard_shapes: dict[str, tuple[int, int]],
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> Tensor:
        # Run processor
        x_encoded_latents = {}
        x_skip = {}

        ## Downscale
        for i in range(0, self.num_hidden - 1):
            src_hidden_name = self._graph_name_hidden[i]
            dst_hidden_name = self._graph_name_hidden[i + 1]

            # Processing at same level
            if self.level_process:
                curr_latent = self.down_level_processor[src_hidden_name](
                    curr_latent,
                    batch_size=batch_size,
                    shard_shapes=shard_shapes[src_hidden_name],
                    model_comm_group=model_comm_group,
                )

            # store latents for skip connections
            x_skip[src_hidden_name] = curr_latent

            # Encode to next hidden level
            x_encoded_latents[src_hidden_name], curr_latent = self._run_mapper(
                self.downscale[src_hidden_name],
                (curr_latent, x[dst_hidden_name]),
                batch_size=batch_size,
                shard_shapes=(shard_shapes[src_hidden_name], shard_shapes[dst_hidden_name]),
                model_comm_group=model_comm_group,
            )

        # Processing hidden-most level
        curr_latent = self.down_level_processor[dst_hidden_name](
            curr_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes[dst_hidden_name],
            model_comm_group=model_comm_group,
        )

        ## Upscale
        for i in range(self.num_hidden - 1, 0, -1):
            src_hidden_name = self._graph_name_hidden[i]
            dst_hidden_name = self._graph_name_hidden[i - 1]

            # Process to next level
            curr_latent = self._run_mapper(
                self.upscale[src_hidden_name],
                (curr_latent, x_encoded_latents[dst_hidden_name]),
                batch_size=batch_size,
                shard_shapes=(shard_shapes[src_hidden_name], shard_shapes[dst_hidden_name]),
                model_comm_group=model_comm_group,
            )

            # Add skip connections
            curr_latent = curr_latent + x_skip[dst_hidden_name]

            # Processing at same level
            if self.level_process:
                curr_latent = self.up_level_processor[dst_hidden_name](
                    curr_latent,
                    batch_size=batch_size,
                    shard_shapes=shard_shapes[dst_hidden_name],
                    model_comm_group=model_comm_group,
                )

        return curr_latent
