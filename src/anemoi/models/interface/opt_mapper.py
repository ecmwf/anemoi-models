# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import uuid
from typing import Optional
from torch.distributed.distributed_c10d import ProcessGroup

import torch
import torch.nn as nn
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from torch_geometric.data import HeteroData

from anemoi.models.preprocessing import Processors
from anemoi.models.layers import processor, mapper
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.interface import AnemoiModelInterface



class OptMapperModel(AnemoiModelEncProcDec):
    """Message passing graph neural network."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        graph_data: HeteroData,
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        model_config : DotDict
            Model configuration
        data_indices : dict
            Data indices
        graph_data : HeteroData
            Graph definition
        """
        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            graph_data=graph_data
        )

        input_dim = self.multi_step * self.num_input_channels + self.latlons_data.shape[1] + self.trainable_data_size

        # Encoder data -> hidden
        self.encoder = mapper.GraphTransformerForwardMapper(
            trainable_size=model_config.model.trainable_parameters.data2hidden,
            sub_graph_edge_attributes=model_config.model.attributes.edges,
            activation=model_config.model.activation,
            num_chunks=1,
            mlp_hidden_ratio=4, 
            num_heads=16,
            in_channels_src=input_dim,
            in_channels_dst=self.latlons_hidden.shape[1] + self.trainable_hidden_size,
            hidden_dim=self.num_channels,
            sub_graph=self._graph_data[(self._graph_name_data, "to", self._graph_name_hidden)],
            src_grid_size=self._data_grid_size,
            dst_grid_size=self._hidden_grid_size,
        )

        # Processor hidden -> hidden
        self.processor = processor.TransformerProcessor(
            activation=model_config.model.activation,
            num_layers=1,
            num_chunks=1,
            mlp_hidden_ratio=4,
            num_heads=16,
            window_size=512,
            dropout_p=0.0,
            num_channels=self.num_channels,
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_hidden)],
            src_grid_size=self._hidden_grid_size,
            dst_grid_size=self._hidden_grid_size,
        )

        # Decoder hidden -> data
        self.decoder = mapper.GraphTransformerBackwardMapper(
            trainable_size=model_config.model.trainable_parameters.hidden2data,
            sub_graph_edge_attributes=model_config.model.attributes.edges,
            activation=model_config.model.activation,
            num_chunks=1,
            mlp_hidden_ratio=4,
            num_heads=16,
            in_channels_src=self.num_channels,
            in_channels_dst=input_dim,
            hidden_dim=self.num_channels,
            out_channels_dst=self.num_output_channels,
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_data)],
            src_grid_size=self._hidden_grid_size,
            dst_grid_size=self._data_grid_size,
        )

        # Instantiation of model output bounding functions (e.g., to ensure outputs like TP are positive definite)
        self.boundings = nn.ModuleList(
            [
                instantiate(cfg, name_to_index=self.data_indices.internal_model.output.name_to_index)
                for cfg in getattr(model_config.model, "bounding", [])
            ]
        )


class OptMapperInterface(AnemoiModelInterface):
    """An interface for Anemoi models.

    This class is a wrapper around the Anemoi model that includes pre-processing and post-processing steps.
    It inherits from the PyTorch Module class.

    Attributes
    ----------
    config : DotDict
        Configuration settings for the model.
    id : str
        A unique identifier for the model instance.
    multi_step : bool
        Whether the model uses multi-step input.
    graph_data : HeteroData
        Graph data for the model.
    statistics : dict
        Statistics for the data.
    metadata : dict
        Metadata for the model.
    data_indices : dict
        Indices for the data.
    pre_processors : Processors
        Pre-processing steps to apply to the data before passing it to the model.
    post_processors : Processors
        Post-processing steps to apply to the model's output.
    model : AnemoiModelEncProcDec
        The underlying Anemoi model.
    """

    def __init__(
        self, *, config: DotDict, graph_data: HeteroData, statistics: dict, data_indices: dict, metadata: dict
    ) -> None:
        super().__init__(
            config=config, graph_data=graph_data, statistics=statistics,
            data_indices=data_indices, metadata=metadata
        )
        self._build_opt_mapper()

    def _build_opt_mapper(self) -> None:
        """Builds the model and pre- and post-processors."""
        # Instantiate the initial condition mapper

        self.opt_mapper = OptMapperModel( 
            model_config=self.config,
            data_indices=self.data_indices,
            graph_data=self.graph_data,
        )
    
    def forward(self, x: torch.Tensor, 
                model_comm_group: Optional[ProcessGroup] = None,
                use_opt_mapper: bool = True) -> torch.Tensor:
        """Forward pass for the model.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Output data.
        """
        if use_opt_mapper:
            x = self.opt_mapper(x, model_comm_group=model_comm_group)
        
        return self.model(x, model_comm_group=model_comm_group)
