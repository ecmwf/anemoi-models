# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations


from torch import nn


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