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

from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.models.base import BaseAnemoiEncProcDecModel

LOGGER = logging.getLogger(__name__)


class AnemoiModelEncProcDec(BaseAnemoiEncProcDecModel):
    """Message passing graph neural network."""

    def instantiate_encoder(self, model_config: DotDict) -> None:
        input_dim = self.multi_step * self.num_input_channels + self.node_attributes.attr_ndims[self._graph_name_data]

        self.encoder = instantiate(
            model_config.model.encoder,
            in_channels_src=input_dim,
            in_channels_dst=self.node_attributes.attr_ndims[self._graph_name_hidden],
            hidden_dim=self.num_channels,
            sub_graph=self._graph_data[(self._graph_name_data, "to", self._graph_name_hidden)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_data],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
        )

    def instantiate_processor(self, model_config: DotDict) -> None:
        self.processor = instantiate(
            model_config.model.processor,
            num_channels=self.num_channels,
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_hidden)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
        )

    def instantiate_decoder(self, model_config: DotDict) -> None:
        input_dim = self.multi_step * self.num_input_channels + self.node_attributes.attr_ndims[self._graph_name_data]

        self.decoder = instantiate(
            model_config.model.decoder,
            in_channels_src=self.num_channels,
            in_channels_dst=input_dim,
            hidden_dim=self.num_channels,
            out_channels_dst=self.num_output_channels,
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_data)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_data],
        )

    def encode(
        self,
        x: tuple[Tensor, Tensor],
        batch_size: int,
        shard_shapes: tuple[int, int],
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[Tensor, Tensor]:
        return self._run_mapper(
            self.encoder, x, batch_size, shard_shapes=shard_shapes, model_comm_group=model_comm_group
        )

    def process(
        self,
        x: Tensor,
        batch_size: int,
        shard_shapes: tuple[int, int],
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> Tensor:
        return self._run_mapper(
            self.processor, x, batch_size, shard_shapes=shard_shapes, model_comm_group=model_comm_group
        )

    def decode(self, x: tuple[Tensor, Tensor], batch_size: int, shard_shapes: tuple[int, int], model_comm_group):
        return self._run_mapper(
            self.decoder, x, batch_size, shard_shapes=shard_shapes, model_comm_group=model_comm_group
        )
