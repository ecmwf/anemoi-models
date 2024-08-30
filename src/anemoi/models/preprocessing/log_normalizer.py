# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
import warnings
from typing import Optional

import numpy as np
import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import BasePreprocessor

LOGGER = logging.getLogger(__name__)


class InputLogNormalizer(BasePreprocessor):
    """Log normalizes input data with a configurable method."""

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        """Initialize the normalizer.

        Parameters
        ----------
        config : DotDict
            configuration object
        statistics : dict
            Data statistics dictionary
        data_indices : dict
            Data indices for input and output variables
        """
        super().__init__(config, statistics, data_indices)

        name_to_index_training_input = self.data_indices.data.input.name_to_index

        name_to_index_training_input = self.data_indices.data.input.name_to_index
        name_to_index_inference_input = self.data_indices.model.input.name_to_index
        name_to_index_training_output = self.data_indices.data.output.name_to_index
        name_to_index_inference_output = self.data_indices.model.output.name_to_index

        self.num_training_input_vars = len(name_to_index_training_input)
        self.num_inference_input_vars = len(name_to_index_inference_input)
        self.num_training_output_vars = len(name_to_index_training_output)
        self.num_inference_output_vars = len(name_to_index_inference_output)

        self._validate_normalization_inputs(name_to_index_training_input,)

        sd_cap = 1.0
        self.norm = {"log": lambda x: torch.log(x + 1),
                     "log-piecewise": lambda x: torch.where(x <= sd_cap, 
                                         x / sd_cap, 
                                         1 + torch.log10(x - sd_cap + 1)),}
        self.norm_inverse = {"log": lambda x: torch.exp(x) - 1,
                             "log-piecewise": lambda x: torch.where(x <= 1,
                                         x * sd_cap,
                                         10**(x - 1) + sd_cap - 1),}

    def _validate_normalization_inputs(self, name_to_index_training_input: dict):
        assert len(self.methods) == sum(len(v) for v in self.method_config.values()), (
            f"Error parsing methods in InputLogNormalizer methods ({len(self.methods)}) "
            f"and entries in config ({sum(len(v) for v in self.method_config)}) do not match."
        )

        assert isinstance(self.methods, dict)
        for name, method in self.methods.items():
            assert name in name_to_index_training_input, f"{name} is not a valid variable name"
            assert method in [
                "log",
                "log-piecewise",
            ], f"{method} is not a valid normalisation method"

    def transform(
        self, x: torch.Tensor, in_place: bool = True,
    ) -> torch.Tensor:
        """Normalizes an input tensor x of shape [..., nvars].

        Normalization done in-place unless specified otherwise.

        The default usecase either assume the full batch tensor or the full input tensor.
        A dataindex is based on the full data can be supplied to choose which variables to normalise.

        Parameters
        ----------
        x : torch.Tensor
            Data to normalize
        in_place : bool, optional
            Normalize in-place, by default True
        data_index : Optional[torch.Tensor], optional
            Normalize only the specified indices, by default None

        Returns
        -------
        torch.Tensor
            _description_
        """
        if not in_place:
            x = x.clone()

        if x.shape[-1] == self.num_training_input_vars:
            index = self.data_indices.data.input.name_to_index
        elif x.shape[-1] == self.num_inference_input_vars:
            index = self.data_indices.model.input.name_to_index
        else:
            raise ValueError(
                f"Input tensor ({x.shape[-1]}) does not match the training "
                f"({self.num_training_input_vars}) or inference shape ({self.num_inference_input_vars})",
            )

        for method in self.method_config.keys():
            for variable in self.method_config[method]:
                x[..., index[variable]] = self.norm[method](
                    x[..., index[variable]]
                )
        return x

    def inverse_transform(
        self, x: torch.Tensor, in_place: bool = True, data_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Denormalizes an input tensor x of shape [..., nvars | nvars_pred].

        Denormalization done in-place unless specified otherwise.

        The default usecase either assume the full batch tensor or the full output tensor.
        A dataindex is based on the full data can be supplied to choose which variables to denormalise.

        Parameters
        ----------
        x : torch.Tensor
            Data to denormalize
        in_place : bool, optional
            Denormalize in-place, by default True
        data_index : Optional[torch.Tensor], optional
            Denormalize only the specified indices, by default None

        Returns
        -------
        torch.Tensor
            Denormalized data
        """
        if not in_place:
            x = x.clone()

        # Replace original nans with nan again
        if x.shape[-1] == self.num_training_output_vars:
            index = self.data_indices.data.output.name_to_index
        elif x.shape[-1] == self.num_inference_output_vars:
            index = self.data_indices.model.output.name_to_index
        else:
            raise ValueError(
                f"Input tensor ({x.shape[-1]}) does not match the training "
                f"({self.num_training_output_vars}) or inference shape ({self.num_inference_output_vars})",
            )

        for method in self.method_config.keys():
            for variable in self.method_config[method]:
                x[..., index[variable]] = self.norm_inverse[method](
                    x[..., index[variable]]
                )
        return x
