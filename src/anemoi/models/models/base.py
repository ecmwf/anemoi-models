# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
from abc import ABC
from abc import abstractmethod
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

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.distributed.shapes import get_shape_shards
from anemoi.models.layers.graph import NamedNodesAttributes

LOGGER = logging.getLogger(__name__)


class BaseAnemoiEncProcDecModel(nn.Module, ABC):
    """Message passing graph neural network."""

    graph_data: HeteroData
    _graph_name_data: str
    _graph_name_hidden: str
    multi_step: int
    num_channels: int
    node_attributes: NamedNodesAttributes
    num_input_channels: int
    num_output_channels: int
    _internal_input_idx: list[int]
    _internal_output_idx: list[int]

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: IndexCollection,
        graph_data: HeteroData,
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        model_config : DotDict
            Model configuration
        data_indices : IndexCollection
            Data indices
        graph_data : HeteroData
            Graph definition
        """
        super().__init__()

        self._graph_data = graph_data
        self._graph_name_data = model_config.graph.data
        self._graph_name_hidden = model_config.graph.hidden

        self._calculate_shapes_and_indices(data_indices)
        self._assert_matching_indices(data_indices)

        self.multi_step = model_config.training.multistep_input
        self.num_channels = model_config.model.num_channels

        self.node_attributes = NamedNodesAttributes(model_config.model.trainable_parameters.hidden, self._graph_data)

        self.instantiate_encoder(model_config)
        self.instantiate_processor(model_config)
        self.instantiate_decoder(model_config)

        self.instantiate_boundings(model_config, data_indices)

    def _calculate_shapes_and_indices(self, data_indices: IndexCollection) -> None:
        self.num_input_channels = len(data_indices.internal_model.input)
        self.num_output_channels = len(data_indices.internal_model.output)
        self._internal_input_idx = data_indices.internal_model.input.prognostic
        self._internal_output_idx = data_indices.internal_model.output.prognostic

    def _assert_matching_indices(self, data_indices: IndexCollection) -> None:
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

    @abstractmethod
    def instantiate_encoder(self, model_config: DotDict) -> None:
        pass

    @abstractmethod
    def instantiate_processor(self, model_config: DotDict) -> None:
        pass

    @abstractmethod
    def instantiate_decoder(self, model_config: DotDict) -> None:
        pass

    def instantiate_boundings(self, model_config: DotDict, data_indices: IndexCollection) -> None:
        # Instantiation of model output bounding functions (e.g., to ensure outputs like TP are positive definite)
        self.boundings = nn.ModuleList(
            [
                instantiate(cfg, name_to_index=data_indices.internal_model.output.name_to_index)
                for cfg in getattr(model_config.model, "bounding", [])
            ]
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

    @abstractmethod
    def encode(
        self,
        x: tuple[Tensor, Tensor],
        batch_size: int,
        shard_shapes: tuple[int, int],
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def process(
        self,
        x: Tensor,
        batch_size: int,
        shard_shapes: tuple[int, int],
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> Tensor:
        pass

    @abstractmethod
    def decode(self, x: tuple[Tensor, Tensor], batch_size: int, shard_shapes: tuple[int, int], model_comm_group):
        pass

    def bound_output(self, x: torch.Tensor) -> torch.Tensor:
        for bounding in self.boundings:
            # bounding performed in the order specified in the config file
            x = bounding(x)

        return x

    def forward(self, x: Tensor, model_comm_group: Optional[ProcessGroup] = None) -> Tensor:
        batch_size = x.shape[0]
        ensemble_size = x.shape[2]

        # add data positional info (lat/lon)
        x_data = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                self.node_attributes(self._graph_name_data, batch_size=batch_size),
            ),
            dim=-1,  # feature dimension
        )

        x_hidden = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)

        # get shard shapes
        shard_shapes_data = get_shape_shards(x_data, 0, model_comm_group)
        shard_shapes_hidden = get_shape_shards(x_hidden, 0, model_comm_group)

        x_data_latent, x_hidden_latent = self.encode(
            (x_data, x_hidden),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_data, shard_shapes_hidden),
            model_comm_group=model_comm_group,
        )

        x_hidden_latent_proc = self.process(x_hidden_latent, batch_size, shard_shapes_hidden, model_comm_group)

        # add skip connection (hidden -> hidden)
        x_hidden_latent_proc = x_hidden_latent_proc + x_hidden_latent

        # Run decoder
        x_out = self.decode(
            (x_hidden_latent_proc, x_data_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hidden, shard_shapes_data),
            model_comm_group=model_comm_group,
        )

        x_out = (
            einops.rearrange(
                x_out,
                "(batch ensemble grid) vars -> batch ensemble grid vars",
                batch=batch_size,
                ensemble=ensemble_size,
            )
            .to(dtype=x.dtype)
            .clone()
        )

        # residual connection (just for the prognostic variables)
        x_out[..., self._internal_output_idx] += x[:, -1, :, :, self._internal_input_idx]

        x_out = self.bound_output(x_out, batch_size, ensemble_size)

        return x_out
