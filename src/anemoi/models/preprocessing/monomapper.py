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
    boxcox_converter,
    expm1_converter,
    inverse_boxcox_converter,
    log1p_converter,
    sqrt_converter,
    square_converter,
)

LOGGER = logging.getLogger(__name__)


class Monomapper(BasePreprocessor, ABC):
    """Remap and convert variables for single variables."""

    supported_methods = {
        method: [f, inv]
        for method, f, inv in zip(
            ["log1p", "sqrt", "boxcox"],
            [log1p_converter, sqrt_converter, boxcox_converter],
            [expm1_converter, square_converter, inverse_boxcox_converter],
        )
    }

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        super().__init__(config, data_indices, statistics)
        self.remappers = []
        self.backmappers = []
        for name, method in self.methods.items():
            method = method or self.default
            if method == "none":
                continue
            elif method in self.supported_methods:
                self.remappers.append(self.supported_methods[method][0])
                self.backmappers.append(self.supported_methods[method][1])
            else:
                raise ValueError(f"Unknown remapping method for {name}: {method}")
        self._create_remapping_indices(statistics)
        self._validate_indices()

    def _validate_indices(self):
        assert (
            len(self.index_training) == len(self.index_inference) <= len(self.remappers)
        ), (
            f"Error creating conversion indices {len(self.index_training)}, "
            f"{len(self.index_inference)}, {len(self.remappers)}"
        )

    def _create_remapping_indices(
        self,
        statistics=None,
    ):
        """Create the parameter indices for remapping."""
        # list for training and inference mode as position of parameters can change
        name_to_index_training = self.data_indices.data.input.name_to_index
        name_to_index_inference = self.data_indices.model.input.name_to_index
        self.num_training_vars = len(name_to_index_training)
        self.num_inference_vars = len(name_to_index_inference)

        (
            self.index_training,
            self.index_inference,
        ) = ([], [])

        # Create parameter indices for remapping variables
        for name in name_to_index_training:
            method = self.methods.get(name, self.default)
            if method == "none":
                continue
            elif method in self.supported_methods:
                self.index_training.append(name_to_index_training[name])
                if name in name_to_index_inference:
                    self.index_inference.append(name_to_index_inference[name])
                else:
                    # this is a forcing variable. It is not in the inference output.
                    self.index_inference.append(None)
                for name_dst in self.method_config[method][name]:
                    assert (
                        name_dst in self.data_indices.internal_data.input.name_to_index
                    ), (
                        f"Trying to remap {name} to {name_dst}, but {name_dst} not a variable. "
                        f"Remap {name} to {name_dst} in config.data.remapped. "
                    )
            else:
                raise ValueError[f"Unknown remapping method for {name}: {method}"]

    def transform(self, x, in_place: bool = True) -> torch.Tensor:
        for method in self.methods.values():
            remapper = self.supported_methods[method][0]
            if x.shape[-1] == self.num_training_vars:
                idx = self.index_training
            elif x.shape[-1] == self.num_inference_vars:
                idx = self.index_inference
            else:
                raise ValueError(
                    f"Input tensor ({x.shape[-1]}) does not match the training "
                    f"({self.num_training_vars}) or inference shape ({self.num_inference_vars})",
                )
            if idx is not None:
                x[..., idx] = remapper(x[..., idx])
        return x

    def inverse_transform(self, x, in_place: bool = True) -> torch.Tensor:
        for method in self.methods.values():
            backmapper = self.supported_methods[method][1]
            if x.shape[-1] == self.num_training_vars:
                idx = self.index_training
            elif x.shape[-1] == self.num_inference_vars:
                idx = self.index_inference
            else:
                raise ValueError(
                    f"Input tensor ({x.shape[-1]}) does not match the training "
                    f"({self.num_training_vars}) or inference shape ({self.num_inference_vars})",
                )
            if idx is not None:
                x[..., idx] = backmapper(x[..., idx])
        return x
