# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import ABC
from typing import Optional

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import BasePreprocessor
from anemoi.models.preprocessing.mappings import (
    log1p_converter,
    boxcox_converter,
    sqrt_converter,
    expm1_converter,
    square_converter,
    inverse_boxcox_converter,
)

LOGGER = logging.getLogger(__name__)


class Remapper(BasePreprocessor, ABC):
    """Remap and convert variables for single variables."""

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        super().__init__(config, data_indices, statistics)
        self.remappers = []
        self.backmappers = []
        self.supported_methods = {
            method: [f, inv]
            for method, f, inv in zip(
                ["log1p", "sqrt", "boxcox"],
                [log1p_converter, sqrt_converter, boxcox_converter],
                [expm1_converter, square_converter, inverse_boxcox_converter],
            )
        }
        for name, method in self.methods.items():
            method = method or self.default
            if method == "none":
                continue
            elif method in self.supported_methods:
                self.remappers.append(self.supported_methods[method][0])
                self.backmappers.append(self.supported_methods[method][1])
            else:
                raise ValueError(f"Unknown remapping method for {name}: {method}")

    def transform(self, x, in_place: bool = True) -> torch.Tensor:
        for name, method in self.methods.items():
            idx = self.data_indices.data.input.name_to_index[name]
            remapper = self.supported_methods[method][0]
            x[..., idx] = remapper(x[..., idx])
        return x

    def inverse_transform(self, x, in_place: bool = True) -> torch.Tensor:
        for name, method in self.methods.items():
            idx = self.data_indices.data.input.name_to_index[name]
            backmapper = self.supported_methods[method][1]
            x[..., idx] = backmapper(x[..., idx])
        return x
