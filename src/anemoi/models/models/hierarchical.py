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
from anemoi.models.layers.processor import BaseProcessor

LOGGER = logging.getLogger(__name__)

from anemoi.models.models import AnemoiModelEncProcDec

class AnemoiModelEncProcDecHierarchical(AnemoiModelEncProcDec):
    """Message passing hierarchical graph neural network."""

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
        config : DotDict
            Job configuration
        data_indices : dict
            Data indices
        graph_data : HeteroData
            Graph definition
        """
        nn.Module.__init__(self)

        self._graph_data = graph_data

        ## Get hidden layers names
        self._graph_name_data = model_config.graph.data
        self._graph_hidden_names = model_config.graph.hidden
        self.num_hidden = len(self._graph_hidden_names)

        # Unpack config for hierarchical graph
        self.level_process = model_config.model.level_process
        
        # hidden_dims is the dimentionality of features at each depth 
        self.hidden_dims = [model_config.model.num_channels*(2**i)-4 for i in range(self.num_hidden)] # first 4 will be for lat-lon positional encoding


        self._calculate_shapes_and_indices(data_indices)
        self._assert_matching_indices(data_indices)
        self.multi_step = model_config.training.multistep_input
        self._define_tensor_sizes(model_config)

        # Create trainable tensors
        self._create_trainable_attributes()

        # Register lat/lon for all nodes
        self._register_latlon("data", "data")
        for hidden in self._graph_hidden_names:
            self._register_latlon(hidden+'_down', hidden)     
            self._register_latlon(hidden+'_up', hidden)     

        input_dim = self.multi_step * self.num_input_channels + self.latlons_data.shape[1] + self.trainable_data_size

        # Encoder data -> hidden
        self.encoder = instantiate(
            model_config.model.encoder,
            in_channels_src=input_dim,
            in_channels_dst=getattr(self, 'latlons_hidden_1_down').shape[1] + self.hidden_dims[0],
            hidden_dim=getattr(self, 'latlons_hidden_1_down').shape[1] + self.hidden_dims[0],
            sub_graph=self._graph_data[(self._graph_name_data, "to", self._graph_hidden_names[0])],
            src_grid_size=self._data_grid_size,
            dst_grid_size=self._hidden_grid_sizes[0],
        )

        # Downscale
        self.downscale = nn.ModuleList()
        for i in range(0, self.num_hidden):
            # Processing at same level
            if self.level_process:
                self.downscale.append(
                    instantiate(
                        model_config.model.processor,
                        num_channels= getattr(self, f'latlons_hidden_{i+1}_down').shape[1] + self.hidden_dims[i],
                        sub_graph=self._graph_data[(self._graph_hidden_names[i], "to", self._graph_hidden_names[i])],
                        src_grid_size=self._hidden_grid_sizes[i],
                        dst_grid_size=self._hidden_grid_sizes[i],
                        num_layers=model_config.model.level_process_num_layers
                        )
                )
            
            # Process to next level
            if i != self.num_hidden-1:
                self.downscale.append(
                    instantiate(
                        model_config.model.encoder,
                        in_channels_src=getattr(self, f'latlons_hidden_{i+2}_down').shape[1] + self.hidden_dims[i],
                        in_channels_dst=getattr(self, f'latlons_hidden_{i+2}_down').shape[1] + self.hidden_dims[i+1],
                        hidden_dim= getattr(self, f'latlons_hidden_{i+2}_down').shape[1] + self.hidden_dims[i+1],
                        sub_graph=self._graph_data[(self._graph_hidden_names[i], "to", self._graph_hidden_names[i+1])],
                        src_grid_size=self._hidden_grid_sizes[i],
                        dst_grid_size=self._hidden_grid_sizes[i+1],
                    )
                )
        
        # Upscale
        self.upscale = nn.ModuleList()
        for i in range(self.num_hidden-1, 0, -1):
            # Processing at same level
            # Skip first otherwise double self message passing at hiddenmost level
            if self.level_process and i != self.num_hidden-1:
                self.upscale.append(
                    instantiate(
                        model_config.model.processor,
                        num_channels=getattr(self, f'latlons_hidden_{i+1}_up').shape[1] + self.hidden_dims[i],
                        sub_graph=self._graph_data[(self._graph_hidden_names[i], "to", self._graph_hidden_names[i])],
                        src_grid_size=self._hidden_grid_sizes[i],
                        dst_grid_size=self._hidden_grid_sizes[i],
                        num_layers=model_config.model.level_process_num_layers
                        )
                )
            
            # Process to next level
            self.upscale.append(
                instantiate(
                    model_config.model.decoder,
                    in_channels_src=getattr(self, f'latlons_hidden_{i+1}_up').shape[1] + self.hidden_dims[i],
                    in_channels_dst=getattr(self, f'latlons_hidden_{i}_up').shape[1] + self.hidden_dims[i-1],
                    hidden_dim=getattr(self, f'latlons_hidden_{i}_up').shape[1] + self.hidden_dims[i],
                    out_channels_dst=getattr(self, f'latlons_hidden_{i}_up').shape[1] + self.hidden_dims[i-1],
                    sub_graph=self._graph_data[(self._graph_hidden_names[i], "to", self._graph_hidden_names[i-1])],
                    src_grid_size=self._hidden_grid_sizes[i],
                    dst_grid_size=self._hidden_grid_sizes[i-1],

                )
            )

        # Decoder hidden -> data
        self.decoder = instantiate(
            model_config.model.decoder,
            in_channels_src=self.hidden_dims[0],
            in_channels_dst=input_dim,
            hidden_dim=getattr(self, f'latlons_hidden_{self.num_hidden}_up').shape[1] + self.hidden_dims[0],
            out_channels_dst=self.num_output_channels,
            sub_graph=self._graph_data[(self._graph_hidden_names[0], "to", self._graph_name_data)],
            src_grid_size=self._hidden_grid_sizes[0],
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
        self.trainable_hidden = nn.ModuleList()

        # Downscale
        for i in range(0, self.num_hidden):
            self.trainable_hidden.append(
                TrainableTensor(
                    trainable_size=self.hidden_dims[i], 
                    tensor_size=self._hidden_grid_sizes[i]
                )
            )
        
        # Upscale
        for i in range(self.num_hidden-2, -1, -1):
            self.trainable_hidden.append(
                TrainableTensor(
                    trainable_size=self.hidden_dims[i], 
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

        # Get all trainable parameters for the hidden layers -> initialisation of each hidden, which becomes trainable bias 
        x_latents = []
        j=0 
        for i in range(1, self.num_hidden + 1): 
            x_latents.append(self.trainable_hidden[j](getattr(self, f'latlons_hidden_{i}_down'), batch_size=batch_size))
            j+=1
                
        for i in range(self.num_hidden-1, 0, -1):
            x_latents.append(self.trainable_hidden[j](getattr(self, f'latlons_hidden_{i}_up'), batch_size=batch_size))
            j+=1
        
        # Get data and hidden shapes for sharding
        shard_shapes_data = get_shape_shards(x_data_latent, 0, model_comm_group)
        shard_shapes_hiddens = []
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
        curr_latent = x_latent   

        ## Downscale
        for i, layer in enumerate(self.downscale):
            
            # Processing at same level
            if isinstance(layer, BaseProcessor):
                curr_latent = layer(
                    curr_latent,
                    batch_size=batch_size,
                    shard_shapes=shard_shapes_hiddens[i],
                    model_comm_group=model_comm_group,
                )

                # store latents for skip connections
                x_latents_down.append(curr_latent)
            
            # Downscale to next layer
            else:
                _, curr_latent = self._run_mapper(
                    layer,
                    (curr_latent, x_latents[(i//2)+1]),
                    batch_size=batch_size,
                    shard_shapes=(shard_shapes_hiddens[i//2], shard_shapes_hiddens[(i//2)+1]),
                    model_comm_group=model_comm_group,
                )

        # Remove hiddenmost latent from skip connection list
        x_latents_down.pop()

        ## Upscale
        for j, layer in enumerate(self.upscale):
            layer_idx = i+j # i+j is cumulative layer index

            # Processing at same level
            if  isinstance(layer, BaseProcessor):
                curr_latent = layer(
                        curr_latent,
                        batch_size=batch_size,
                        shard_shapes=shard_shapes_hiddens[layer_idx//2],
                        model_comm_group=model_comm_group,
                    )
                
                curr_latent += x_latents_down.pop()

            # Process to next level
            else:
                curr_latent = self._run_mapper(
                    layer,
                    (curr_latent, x_latents[(layer_idx//2)+1]),
                    batch_size=batch_size,
                    shard_shapes=(shard_shapes_hiddens[layer_idx//2], shard_shapes_hiddens[(layer_idx//2)+1]),
                    model_comm_group=model_comm_group,
                )

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

        for bounding in self.boundings:
            # bounding performed in the order specified in the config file
            x_out = bounding(x_out)

        return x_out