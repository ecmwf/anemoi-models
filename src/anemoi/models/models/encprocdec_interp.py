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

import einops
import torch
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.distributed.shapes import get_shape_shards

LOGGER = logging.getLogger(__name__)


class AnemoiModelEncProcDecInterp(AnemoiModelEncProcDec):
    """Message passing graph neural network."""

    def forward(self, x: Tensor, target_forcing: torch.Tensor, model_comm_group: Optional[ProcessGroup] = None) -> Tensor:
        batch_size = x.shape[0]
        ensemble_size = x.shape[2]

        # add data positional info (lat/lon)
        x_data = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                einops.rearrange(target_forcing, "batch ensemble grid vars -> (batch ensemble grid) (vars)"),
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
        x_out[..., self._internal_output_idx] += x[:, 0, :, :, self._internal_input_idx]

        x_out = self.bound_output(x_out) #, batch_size, ensemble_size)

        return x_out
