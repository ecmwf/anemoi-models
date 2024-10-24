# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import torch
from torch import nn

from anemoi.models.data_indices.tensor import InputTensorIndex
from anemoi.models.preprocessing.normalizer import InputNormalizer

class BaseBounding(nn.Module, ABC):
    """Abstract base class for bounding strategies.

    This class defines an interface for bounding strategies which are used to apply a specific
    restriction to the predictions of a model.
    """

    def __init__(
        self,
        *,
        variables: list[str],
        name_to_index: dict,
    ) -> None:
        super().__init__()

        self.name_to_index = name_to_index
        self.variables = variables
        self.data_index = self._create_index(variables=self.variables)

    def _create_index(self, variables: list[str]) -> InputTensorIndex:
        return InputTensorIndex(includes=variables, excludes=[], name_to_index=self.name_to_index)._only

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the bounding to the predictions.

        Parameters
        ----------
        x : torch.Tensor
            The tensor containing the predictions that will be bounded.

        Returns
        -------
        torch.Tensor
        A tensor with the bounding applied.
        """
        pass


class ReluBounding(BaseBounding):
    """Initializes the bounding with a ReLU activation / zero clamping."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[..., self.data_index] = torch.nn.functional.relu(x[..., self.data_index])
        return x


class HardtanhBounding(BaseBounding):
    """Initializes the bounding with specified minimum and maximum values for bounding.

    Parameters
    ----------
    variables : list[str]
        A list of strings representing the variables that will be bounded.
    name_to_index : dict
        A dictionary mapping the variable names to their corresponding indices.
    min_val : float
        The minimum value for the HardTanh activation.
    max_val : float
        The maximum value for the HardTanh activation.
    """

    def __init__(self, *, variables: list[str], name_to_index: dict, min_val: float, max_val: float) -> None:
        super().__init__(variables=variables, name_to_index=name_to_index)
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[..., self.data_index] = torch.nn.functional.hardtanh(
            x[..., self.data_index], min_val=self.min_val, max_val=self.max_val
        )
        return x


class FractionBounding(HardtanhBounding):
    """Initializes the FractionBounding with specified parameters.

    Parameters
    ----------
    variables : list[str]
        A list of strings representing the variables that will be bounded.
    name_to_index : dict
        A dictionary mapping the variable names to their corresponding indices.
    min_val : float
        The minimum value for the HardTanh activation.
    max_val : float
        The maximum value for the HardTanh activation.
    total_var : str
        A string representing a variable from which a secondary variable is derived. For
        example, in the case of convective precipitation (Cp), total_var = Tp (total precipitation).
    """

    def __init__(
        self, *, variables: list[str], name_to_index: dict, min_val: float, max_val: float, total_var: str
    ) -> None:
        super().__init__(variables=variables, name_to_index=name_to_index, min_val=min_val, max_val=max_val)
        self.total_variable = self._create_index(variables=[total_var])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the HardTanh bounding  to the data_index variables
        x = super().forward(x)
        # Calculate the fraction of the total variable
        x[..., self.data_index] *= x[..., self.total_variable]
        return x


class MaskBounding(BaseBounding):
    """Initializes the MaskBounding with a mask variable, threshold, and mask type.

    Parameters
    ----------
    variables : list[str]
        A list of strings representing the variables that will be masked.
    name_to_index : dict
        A dictionary mapping the variable names to their corresponding indices.
    mask_var : str
        The name of the variable on which the mask is based.
    trs_val : float
        The threshold value for creating the mask.
    custom_value : float, optional
        A custom value to assign to the masked regions. If not provided, default behavior is to set masked regions to zero.
    mask_type : str, optional
        The type of mask to apply: '>=' (default) or '<='. Determines whether the mask is active
        when the variable is greater than or equal to the threshold or less than or equal to the threshold.
    """

    def __init__(
        self, *, variables: list[str], name_to_index: dict, mask_var: str, trs_val: float, custom_value: float = 0.0, mask_type: str = ">="
    ) -> None:
        super().__init__(variables=variables, name_to_index=name_to_index)
        self.mask_var = self._create_index(variables=[mask_var])  # Create index for the mask variable
        self.trs_val = trs_val  # Threshold value
        self.custom_value = custom_value  # Custom value for masked regions
        self.mask_type = mask_type  # Mask type, either '>=' or '<='

        # Validate mask_type input
        if self.mask_type not in {">=", "<="}:
            raise ValueError("mask_type must be either '>=' or '<='.")

        # Ensure custom_value is a float or int
        if not isinstance(self.custom_value, (float, int)):
            raise ValueError("custom_value must be a float or int")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Generate the mask based on the mask_type
        if self.mask_type == ">=":
            mask = (x[..., self.mask_var] >= self.trs_val).float()
        elif self.mask_type == "<=":
            #mask = (InputNormalizer.inverse_transform(x=x[..., self.mask_var], in_place=False) <= self.trs_val).float()
            mask = (x[..., self.mask_var] <= self.trs_val).float()

        # Ensure mask is the same data type as the input tensor
        mask = mask.to(x.dtype)

        # Apply the mask to the dependent variables (self.data_index)
        # Retain values where mask is 1, and set custom_value where mask is 0
        x[..., self.data_index] = mask * x[..., self.data_index] + (1 - mask) * self.custom_value

        return x
