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
from anemoi.models.layers.graph import TrainableTensor

LOGGER = logging.getLogger(__name__)


class AnemoiModelEncProcDec(nn.Module):
    """Message passing graph neural network."""

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
 
        self._graph_name_data = config.graph.data
        self._graph_name_hidden = config.graph.hidden

        self._calculate_shapes_and_indices(data_indices)
        self._assert_matching_indices(data_indices)

        self.multi_step = config.training.multistep_input

        self._define_tensor_sizes(config)

        # Create trainable tensors
        self._create_trainable_attributes()

        # Register lat/lon of nodes
        self._register_latlon("data", self._graph_name_data)
        self._register_latlon("hidden", self._graph_name_hidden)

        self.num_channels = config.model.num_channels

        input_dim = self.multi_step * self.num_input_channels + self.latlons_data.shape[1] + self.trainable_data_size

        # Encoder data -> hidden
        self.encoder = instantiate(
            config.model.encoder,
            in_channels_src=input_dim,
            in_channels_dst=self.latlons_hidden.shape[1] + self.trainable_hidden_size,
            hidden_dim=self.num_channels,
            sub_graph=self._graph_data[(self._graph_name_data, "to", self._graph_name_hidden)],
            src_grid_size=self._data_grid_size,
            dst_grid_size=self._hidden_grid_size,
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
        self.decoder = instantiate(
            config.model.decoder,
            in_channels_src=self.num_channels,
            in_channels_dst=input_dim,
            hidden_dim=self.num_channels,
            out_channels_dst=self.num_output_channels,
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_data)],
            src_grid_size=self._hidden_grid_size,
            dst_grid_size=self._data_grid_size,
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

    def _define_tensor_sizes(self, config: DotDict) -> None:
        self._data_grid_size = self._graph_data[self._graph_name_data].num_nodes
        self._hidden_grid_size = self._graph_data[self._graph_name_hidden].num_nodes

        self.trainable_data_size = config.model.trainable_parameters.data
        self.trainable_hidden_size = config.model.trainable_parameters.hidden

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

    def _create_trainable_attributes(self) -> None:
        """Create all trainable attributes."""
        self.trainable_data = TrainableTensor(trainable_size=self.trainable_data_size, tensor_size=self._data_grid_size)
        self.trainable_hidden = TrainableTensor(
            trainable_size=self.trainable_hidden_size, tensor_size=self._hidden_grid_size
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
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                self.trainable_data(self.latlons_data, batch_size=batch_size),
            ),
            dim=-1,  # feature dimension
        )

        x_hidden_latent = self.trainable_hidden(self.latlons_hidden, batch_size=batch_size)

        # get shard shapes
        shard_shapes_data = get_shape_shards(x_data_latent, 0, model_comm_group)
        shard_shapes_hidden = get_shape_shards(x_hidden_latent, 0, model_comm_group)

        # Run encoder
        x_data_latent, x_latent = self._run_mapper(
            self.encoder,
            (x_data_latent, x_hidden_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_data, shard_shapes_hidden),
            model_comm_group=model_comm_group,
        )

        x_latent_proc = self.processor(
            x_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        # add skip connection (hidden -> hidden)
        x_latent_proc = x_latent_proc + x_latent

        # Run decoder
        x_out = self._run_mapper(
            self.decoder,
            (x_latent_proc, x_data_latent),
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
        return x_out


class AnemoiModelEncProcDecHierachical(AnemoiModelEncProcDec):
    """Message passing hierarchical graph neural network."""

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

        self._graph_data = graph_data
        self._graph_name_data = config.graph.data
        self.num_hidden = config.graph.num_hidden_layers
        self.level_process = config.graph.lebel_process

        ## Hidden layers
        self._graph_hidden_names = [f'hidden_{i}' for i in range(self.num_hidden)]

        self._calculate_shapes_and_indices(data_indices)
        self._assert_matching_indices(data_indices)

        self.multi_step = config.training.multistep_input

        self._define_tensor_sizes(config)

        # Create trainable tensors
        self._create_trainable_attributes()

        # Register lat/lon of nodes
        self._register_latlon("data", self._graph_name_data)
        self._latlons = []
        for hidden in self._graph_hidden_names:
            self._register_latlon(hidden, hidden)     
            self._latlons.append(hidden)

        self.num_channels = config.model.num_channels
        input_dim = self.multi_step * self.num_input_channels + self.latlons_data.shape[1] + self.trainable_data_size

        # Encoder data -> hidden
        self.encoder = instantiate(
            config.model.encoder,
            in_channels_src=input_dim,
            in_channels_dst=self._latlons[0].shape[1] + self.trainable_hidden_sizes[0],
            hidden_dim=self.num_channels[0],
            sub_graph=self._graph_data[(self._graph_name_data, "to", self._graph_hidden_names[0])],
            src_grid_size=self._data_grid_size,
            dst_grid_size=self._hidden_grid_sizes[0],
        )

        # Downscale
        self.downscale = []
        for i in range(1, self.num_hidden-1):
            # Processing at same level
            if self.level_process:
                self.downscale.append(
                    instantiate(
                        config.model.processor,
                        num_channels=self.num_channels[i],
                        sub_graph=self._graph_data[(self._graph_hidden_names[i], "to", self._graph_hidden_names[i])],
                        src_grid_size=self._hidden_grid_sizes[i],
                        dst_grid_size=self._hidden_grid_sizes[i],
                        )
                )
            
            # Process to next level
            self.downscale.append(
                instantiate(
                    config.model.processor,
                    num_channels=self.num_channels[i+1],
                    sub_graph=self._graph_data[(self._graph_hidden_names[i], "to", self._graph_hidden_names[i+1])],
                    src_grid_size=self._hidden_grid_sizes[i],
                    dst_grid_size=self._hidden_grid_sizes[i+1],
                )
            )

        # Upscale
        self.upscale = []
        for i in range(self.num_hidden-1, 0, -1):
            # Processing at same level
            # Skip first otherwise double self message passing at hiddenmost level
            if (self._graph_hidden_names[i], "to", self._graph_hidden_names[i]) in self._graph_data and i !=len(self._graph_hidden_names)-1:
                self.upscale.append(
                    instantiate(
                        config.model.processor,
                        num_channels=self.num_channels[i],
                        sub_graph=self._graph_data[(self._graph_hidden_names[i], "to", self._graph_hidden_names[i])],
                        src_grid_size=self._hidden_grid_sizes[i],
                        dst_grid_size=self._hidden_grid_sizes[i],
                        )
                )
            
            # Process to next level
            self.upscale.append(
                instantiate(
                    config.model.processor,
                    num_channels=self.num_channels[i-1],
                    sub_graph=self._graph_data[(self._graph_hidden_names[i], "to", self._graph_hidden_names[i-1])],
                    src_grid_size=self._hidden_grid_sizes[i],
                    dst_grid_size=self._hidden_grid_sizes[i-1],
                )
            )

        # Decoder hidden -> data
        self.decoder = instantiate(
            config.model.decoder,
            in_channels_src=self.num_channels[0],
            in_channels_dst=input_dim,
            hidden_dim=self.num_channels[0],
            out_channels_dst=self.num_output_channels,
            sub_graph=self._graph_data[(self._graph_hidden_names[0], "to", self._graph_name_data)],
            src_grid_size=self.self._hidden_grid_sizes[0],
            dst_grid_size=self._data_grid_size,
        )
 
    def _define_tensor_sizes(self, config: DotDict) -> None:

        # Grid sizes
        self._data_grid_size = self._graph_data[self._graph_name_data].num_nodes
        self._hidden_grid_sizes = []
        for hidden in self._graph_hidden_names:
            self._hidden_grid_sizes.append(self._graph_data[hidden].num_nodes)

        # trainable sizes
        self.trainable_data_size = config.model.trainable_parameters.data
        self.trainable_hidden_size = config.model.trainable_parameters.hidden

    def _create_trainable_attributes(self) -> None:
        """Create all trainable attributes."""
        self.trainable_data = TrainableTensor(trainable_size=self.trainable_data_size, tensor_size=self._data_grid_size)
        
        self.trainable_hidden = [] 
        # Downscale
        for i in range(0, self.num_hidden):
            self.trainable_hidden.append(
                TrainableTensor(
                    trainable_size=self.trainable_hidden_size, 
                    tensor_size=self._hidden_grid_sizes[i]
                )
            )
        
        # Upscale
        for i in range(len(self.num_hidden)-1, 0):
            self.trainable_hidden.append(
                TrainableTensor(
                    trainable_size=self.trainable_hidden_size, 
                    tensor_size=self._hidden_grid_sizes[i]
                )
            )
        
    def forward(self, x: Tensor, model_comm_group: Optional[ProcessGroup] = None) -> Tensor:
        batch_size = x.shape[0]
        ensemble_size = x.shape[2]

        # add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                self.trainable_data(self.latlons_data, batch_size=batch_size),
            ),
            dim=-1,  # feature dimension
        )

        x_latents = []
        j=0 # iterator for trainable hidden
        for i in range(self.num_hidden): 
            x_latents.append(self.trainable_hidden[j](self._latlons[i], batch_size=batch_size))
            j+=1
        
        for i in range(self.num_hidden-1, 0):
            x_latents.append(self.trainable_hidden[j](self._latlons[i], batch_size=batch_size))
            j+=1

        # get shard shapes
        shard_shapes_data = get_shape_shards(x_data_latent, 0, model_comm_group)

        shard_shapes_hiddens = []
        j=0 # iterator for trainable hidden
        for x_latent in x_latents: 
            shard_shapes_hiddens.append(get_shape_shards(x_latent, 0, model_comm_group))
        
        # Run encoder
        x_data_latent, x_latent = self._run_mapper(
            self.encoder,
            (x_data_latent, x_latents[0]),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_data, shard_shapes_hiddens[0]),
            model_comm_group=model_comm_group,
        )

        # Run processor
        x_latents_down = []
        skip_connection_idx = 2 if self.level_process else 1
        curr_latent = x_latent

        ## Downscale
        for i,layer in enumerate(self.downscale):
            curr_latent = layer(
                curr_latent,
                batch_size=batch_size,
                shard_shapes=shard_shapes_hiddens[i],
                model_comm_group=model_comm_group,
            )

            # store latents for skip connections
            if i % skip_connection_idx:
                x_latents_down.append(curr_latent)

        # Remove hiddenmost latent from skip connection list
        x_latents_down.pop()

        ## Upscale
        for j, layer in enumerate(self.upscale):
            curr_latent = layer(
                    curr_latent,
                    batch_size=batch_size,
                    shard_shapes=shard_shapes_hiddens[i+j], #Cumulative layer list
                    model_comm_group=model_comm_group,
                )
            # Add skip connection
            if j % skip_connection_idx:
                curr_latent += x_latents_down.pop()

        # Run decoder
        x_out = self._run_mapper(
            self.decoder,
            (curr_latent, x_data_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hiddens[-1], shard_shapes_data),
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
        return x_out
