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

        self.norm = {"log": self.log_transform,
                     "log10-piecewise": self.log10_piecewise_transform,
                     "log-piecewise": self.log_piecewise_transform,
                     "snow-transform": self.snow_transform,}
        
        # self.sd_cap = 0.9
        # self.sd_cap = 0.5
        # self.mask_locations = None
        # sd_max = self.log_piecewise_transform(torch.Tensor([10.]), self.sd_cap)
        sd_max = self.log_transform(torch.Tensor([10.*1000]))
        self.register_buffer("sd_max", sd_max, persistent=True)

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
                "log10-piecewise",
                "log-piecewise",
                "snow-transform",
            ], f"{method} is not a valid normalisation method"

    def log_transform(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Apply log transformation to input tensor x.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        inverse : bool, optional
            Inverse transformation, by default False

        Returns
        -------
        torch.Tensor
            Transformed tensor
        """
        if inverse:
            return torch.expm1(x)
        return torch.log1p(x)
    
    def log10_piecewise_transform(self, x: torch.Tensor, x_cap: float = 0.9, inverse: bool = False) -> torch.Tensor:
        """Apply piecewise log transformation to input tensor x.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        inverse : bool, optional
            Inverse transformation, by default False

        Returns
        -------
        torch.Tensor
            Transformed tensor
        """
        # sd_cap = 0.2
        if inverse:
            return torch.where(x <= 1, x * x_cap, 10**(x - 1) + x_cap - 1)
        return torch.where(x <= x_cap, x / x_cap, 1 + torch.log10(x - x_cap + 1))
    
    def log_piecewise_transform(self, x: torch.Tensor, x_cap: float = 0.9, inverse: bool = False) -> torch.Tensor:
        """Apply piecewise log transformation to input tensor x.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        inverse : bool, optional
            Inverse transformation, by default False

        Returns
        -------
        torch.Tensor
            Transformed tensor
        """
        # sd_cap = 0.2
        if inverse:
            return torch.where(x <= 1, x * x_cap, torch.expm1(x - 1) + x_cap)
        return torch.where(x <= x_cap, x / x_cap, 1 + torch.log1p(x - x_cap))
    
    # def snow_transform(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor: 
    #     if inverse:
    #         x = x * self.sd_max
    #         x = torch.where(x <= -1.0, self.sd_max, x)
    #         x = self.log_piecewise_transform(x, self.sd_cap, inverse=True)
    #         return torch.clip(x, 0.0, 10.0)
    #     x = self.log_piecewise_transform(x, self.sd_cap)
    #     x = torch.where(x >= self.sd_max, -1.0, x)
    #     return x / self.sd_max
    # def snow_transform(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor: 
    #     if inverse:
    #         x = x * self.sd_max
    #         x = self.log_piecewise_transform(x, self.sd_cap, inverse=True)
    #         # x = torch.clip(x, 0., 10.)
    #         return x
    #     x = self.log_piecewise_transform(x, self.sd_cap)
    #     return x / self.sd_max
    # def snow_transform(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor: 
    #     LOGGER.debug(f"transform: x shape {x.shape}, {torch.argwhere(x==10.0).shape}")
    #     if inverse:
    #         x = 10**(x - 1)
    #         x[self.mask_locations] = 10.0
    #         return x
    #     mask_locations = torch.argwhere(x==10.0)
    #     self.register_buffer("mask_locations", mask_locations, persistent=True)
    #     #idx = [slice(0, 1)] * (x.ndim - 2) + [slice(None), slice(None)]
    #     #self.mask_locations = torch.where(x[idx].squeeze() >= 10.0)
    #     x[self.mask_locations] = 0.0
    #     return torch.log10(x + 1.0)
    def snow_transform(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor: 
        if inverse:
            x = x * self.sd_max
            x = self.log_transform(x, inverse=True)/1000.
            return x
        x = self.log_transform(x*1000.)
        return x / self.sd_max

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
        self, x: torch.Tensor, in_place: bool = True,
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
                x[..., index[variable]] = self.norm[method](
                    x[..., index[variable]], inverse=True
                )
        return x
