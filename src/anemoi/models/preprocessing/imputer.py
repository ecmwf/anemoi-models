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

LOGGER = logging.getLogger(__name__)


class BaseImputer(BasePreprocessor, ABC):
    """Base class for Imputers."""

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        """Initialize the imputer.

        Parameters
        ----------
        config : DotDict
            configuration object of the processor
        data_indices : IndexCollection
            Data indices for input and output variables
        statistics : dict
            Data statistics dictionary
        """
        super().__init__(config, data_indices, statistics)

        self.nan_locations = None

    def _validate_indices(self):
        assert len(self.index_training_input) == len(self.index_inference_input) <= len(self.replacement), (
            f"Error creating imputation indices {len(self.index_training_input)}, "
            f"{len(self.index_inference_input)}, {len(self.replacement)}"
        )
        assert len(self.index_training_output) == len(self.index_inference_output) <= len(self.replacement), (
            f"Error creating imputation indices {len(self.index_training_output)}, "
            f"{len(self.index_inference_output)}, {len(self.replacement)}"
        )

    def _create_imputation_indices(
        self,
        statistics=None,
    ):
        """Create the indices for imputation."""
        name_to_index_training_input = self.data_indices.data.input.name_to_index
        name_to_index_inference_input = self.data_indices.model.input.name_to_index
        name_to_index_training_output = self.data_indices.data.output.name_to_index
        name_to_index_inference_output = self.data_indices.model.output.name_to_index

        self.num_training_input_vars = len(name_to_index_training_input)
        self.num_inference_input_vars = len(name_to_index_inference_input)
        self.num_training_output_vars = len(name_to_index_training_output)
        self.num_inference_output_vars = len(name_to_index_inference_output)

        (
            self.index_training_input,
            self.index_inference_input,
            self.index_training_output,
            self.index_inference_output,
            self.replacement,
        ) = ([], [], [], [], [])

        # Create indices for imputation
        for name in name_to_index_training_input:

            method = self.methods.get(name, self.default)
            if method == "none":
                LOGGER.debug(f"Imputer: skipping {name} as no imputation method is specified")
                continue

            self.index_training_input.append(name_to_index_training_input[name])
            self.index_training_output.append(name_to_index_training_output.get(name, None))
            self.index_inference_input.append(name_to_index_inference_input.get(name, None))
            self.index_inference_output.append(name_to_index_inference_output.get(name, None))

            if statistics is None:
                self.replacement.append(method)
            elif isinstance(statistics, dict):
                assert method in statistics, f"{method} is not a method in the statistics metadata"
                self.replacement.append(statistics[method][name_to_index_training_input[name]])
            else:
                raise TypeError(f"Statistics {type(statistics)} is optional and not a dictionary")

            LOGGER.debug(f"Imputer: replacing NaNs in {name} with value {self.replacement[-1]}")

    def _expand_subset_mask(self, x: torch.Tensor, idx_src: int) -> torch.Tensor:
        """Expand the subset of the mask to the correct shape."""
        return self.nan_locations[:, idx_src].expand(*x.shape[:-2], -1)

    def transform(self, x: torch.Tensor, in_place: bool = True) -> torch.Tensor:
        """Impute missing values in the input tensor."""
        if not in_place:
            x = x.clone()

        # Initilialize mask once
        if self.nan_locations is None:
            # The mask is only saved for the last two dimensions (grid, variable)
            idx = [slice(0, 1)] * (x.ndim - 2) + [slice(None), slice(None)]
            self.nan_locations = torch.isnan(x[idx].squeeze())

        # Choose correct index based on number of variables
        if x.shape[-1] == self.num_training_input_vars:
            index = self.index_training_input
        elif x.shape[-1] == self.num_inference_input_vars:
            index = self.index_inference_input
        else:
            raise ValueError(
                f"Input tensor ({x.shape[-1]}) does not match the training "
                f"({self.num_training_input_vars}) or inference shape ({self.num_inference_input_vars})",
            )

        # Replace values
        for idx_src, (idx_dst, value) in zip(self.index_training_input, zip(index, self.replacement)):
            if idx_dst is not None:
                x[..., idx_dst][self._expand_subset_mask(x, idx_src)] = value
        return x

    def inverse_transform(self, x: torch.Tensor, in_place: bool = True) -> torch.Tensor:
        """Impute missing values in the input tensor."""
        if not in_place:
            x = x.clone()

        # Replace original nans with nan again
        if x.shape[-1] == self.num_training_output_vars:
            index = self.index_training_output
        elif x.shape[-1] == self.num_inference_output_vars:
            index = self.index_inference_output
        else:
            raise ValueError(
                f"Input tensor ({x.shape[-1]}) does not match the training "
                f"({self.num_training_output_vars}) or inference shape ({self.num_inference_output_vars})",
            )

        # Replace values
        for idx_src, idx_dst in zip(self.index_training_input, index):
            if idx_dst is not None:
                x[..., idx_dst][self._expand_subset_mask(x, idx_src)] = torch.nan
        return x


class InputImputer(BaseImputer):
    """Imputes missing values using the statistics supplied.

    Expects the config to have keys corresponding to available statistics
    and values as lists of variables to impute.:
    ```
    default: "none"
    mean:
        - y
    maximum:
        - x
    minimum:
        - q
    ```
    """

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        super().__init__(config, data_indices, statistics)

        self._create_imputation_indices(statistics)

        self._validate_indices()


class ConstantImputer(BaseImputer):
    """Imputes missing values using the constant value.

    Expects the config to have keys corresponding to available statistics
    and values as lists of variables to impute.:
    ```
    default: "none"
    1:
        - y
    5.0:
        - x
    3.14:
        - q
    ```
    """

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        super().__init__(config, data_indices, statistics)

        self._create_imputation_indices()

        self._validate_indices()
