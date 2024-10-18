# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from torch import Tensor
from torch import nn
from torch.utils.checkpoint import checkpoint


class CheckpointWrapper(nn.Module):
    """Wrapper for checkpointing a module."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return checkpoint(self.module, *args, **kwargs, use_reentrant=False)


class AutocastLayerNorm(nn.LayerNorm):
    """LayerNorm that casts the output back to the input type."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """Forward with explicit autocast back to the input type.

        This casts the output to (b)float16 (instead of float32) when we run in mixed
        precision.
        """
        return super().forward(x).type_as(x)

#takes a resolution and calculates the number of grid points
# e.g. o32 -> 5248, o96 -> 40320
def calculate_seq_len(resolution: str):
    grid_type=resolution[0] # e.g. 'o'
    num_lat_lines=int(resolution[1:]) # e.g. 32
    accum=0
    if (grid_type.lower() == 'o'):
        # algorithm from https://confluence.ecmwf.int/display/FCST/Introducing+the+octahedral+reduced+Gaussian+grid
        for i in range(1,num_lat_lines+1):
            accum += (4 * i) + 16
        result = accum * 2 # above was just pole -> equator, double for whole globe
    else:
        ValueError("Only octahedral (reduced) Gaussian grid, 'o', implemented")
    return result