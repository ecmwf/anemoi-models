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
from typing import Optional

import torch
from torch import nn

from anemoi.models.data_indices.tensor import InputTensorIndex


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
        statistics: Optional[dict] = None,
        name_to_index_stats: Optional[dict] = None,
    ) -> None:
        """Initializes the bounding strategy.

        Parameters
        ----------
        variables : list[str]
            A list of strings representing the variables that will be bounded.
        name_to_index : dict
            A dictionary mapping the variable names to their corresponding indices.
        statistics : dict, optional
            A dictionary containing the statistics of the variables.
        name_to_index_stats : dict, optional
            A dictionary mapping the variable names to their corresponding indices in the statistics dictionary
        """
        super().__init__()

        self.name_to_index = name_to_index
        self.variables = variables
        self.data_index = self._create_index(variables=self.variables)
        self.statistics = statistics
        self.name_to_index_stats = name_to_index_stats

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


class NormalizedReluBounding(BaseBounding):
    """Bounding with a ReLU activation with custom normliazed value."""

    def __init__(
        self,
        *,
        variables: list[str],
        name_to_index: dict,
        min_val: list[float],
        normalizer: str,
        statistics: dict,
        name_to_index_stats: dict,
    ) -> None:
        """Initializes the NormalizedReluBounding with specified minimum values for bounding.

        Parameters
        ----------
        variables : list[str]
            A list of strings representing the variables that will be bounded.
        name_to_index : dict
            A dictionary mapping the variable names to their corresponding indices.
        statistics : dict
            A dictionary containing the statistics of the variables.
        min_val : list[float]
            The minimum value for the ReLU activation.
        normalizer : str
            The type of normalizer to apply: 'mean-std'.
        name_to_index_stats : dict
            A dictionary mapping the variable names to their corresponding indices in the statistics dictionary
        """

        super().__init__(
            variables=variables,
            name_to_index=name_to_index,
            statistics=statistics,
            name_to_index_stats=name_to_index_stats,
        )
        self.min_val = min_val
        self.normalizer = normalizer

        # Validate mask_type input
        if self.normalizer not in {"mean-std"}:
            raise ValueError("normalizer must be 'mean-std'.")

        self.norm_min_val = torch.zeros(len(variables))
        for ii, variable in enumerate(variables):
            if self.normalizer == "mean-std":
                self.norm_min_val[ii] = min_val[ii] - self.statistics["mean"][self.name_to_index_stats[variable]]
                self.norm_min_val[ii] *= 1.0 / self.statistics["stdev"][self.name_to_index_stats[variable]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.norm_min_val = self.norm_min_val.to(x.device)
        x[..., self.data_index] = (
            torch.nn.functional.relu(x[..., self.data_index] - self.norm_min_val) + self.norm_min_val
        )
        return x


class HardtanhBounding(BaseBounding):
    """Bounding with a HardTanh activation."""

    def __init__(
        self,
        *,
        variables: list[str],
        name_to_index: dict,
        min_val: float,
        max_val: float,
        statistics: Optional[dict] = None,
        name_to_index_stats: Optional[dict] = None,
    ) -> None:
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
        statistics : dict
            A dictionary containing the statistics of the variables.
        name_to_index_stats : dict
            A dictionary mapping the variable names to their corresponding indices in the statistics dictionary
        """
        super().__init__(variables=variables, name_to_index=name_to_index)
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[..., self.data_index] = torch.nn.functional.hardtanh(
            x[..., self.data_index], min_val=self.min_val, max_val=self.max_val
        )
        return x


class FractionBounding(HardtanhBounding):
    """Bounding with a HardTanh activation and a fraction of a total variable."""

    def __init__(
        self,
        *,
        variables: list[str],
        name_to_index: dict,
        min_val: float,
        max_val: float,
        total_var: str,
        statistics: Optional[dict] = None,
        name_to_index_stats: Optional[dict] = None,
    ) -> None:
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
        statistics : dict
            A dictionary containing the statistics of the variables.
        name_to_index_stats : dict
            A dictionary mapping the variable names to their corresponding indices in the statistics dictionary
        """
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
    statistics : dict
        A dictionary containing the statistics of the variables.
    name_to_index_stats : dict
        A dictionary mapping the variable names to their corresponding indices in the statistics dictionary
    custom_value : float, optional
        A custom value to assign to the masked regions. If not provided, default behavior is to set masked regions to zero.
    mask_type : str, optional
        The type of mask to apply: '>=' (default) or '<='. Determines whether the mask is active
        when the variable is greater than or equal to the threshold or less than or equal to the threshold.
    """

    def __init__(
        self,
        *,
        variables: list[str],
        name_to_index: dict,
        mask_var: str,
        trs_val: float,
        statistics: Optional[dict] = None,
        name_to_index_stats: Optional[dict] = None,
        custom_value: float = 0.0,
        mask_type: str = ">=",
    ) -> None:
        super().__init__(variables=variables, name_to_index=name_to_index)
        self.mask_var = self._create_index(variables=[mask_var])  # Create index for the mask variable
        self.trs_val = trs_val  # Threshold value
        self.custom_value = custom_value  # Custom value for masked regions
        self.mask_type = mask_type  # Mask type, either '>=' or '<='

        # example
        # mask_var: siconc
        # trs_val: applied to siconc
        # variables: si_velo
        # custom_value: applied to si_velo

        self.statistics = statistics

        # Validate mask_type input
        if self.mask_type not in {">=", "<="}:
            raise ValueError("mask_type must be either '>=' or '<='.")

        # Ensure custom_value is a float or int
        if not isinstance(self.custom_value, (float, int)):
            raise ValueError("custom_value must be a float or int")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply a differentiable sigmoid approximation for the mask
        if self.mask_type == ">=":
            mask = torch.sigmoid(10 * (x[..., self.mask_var] - self.trs_val))
        elif self.mask_type == "<=":
            mask = torch.sigmoid(-10 * (x[..., self.mask_var] - self.trs_val))

        # Ensure mask is the same data type as the input tensor
        mask = mask.to(x.dtype)

        # Apply the mask to the dependent variables (self.data_index)
        # Retain values where mask is close to 1, and set custom_value where mask is close to 0
        x[..., self.data_index] = mask * x[..., self.data_index] + (1 - mask) * self.custom_value

        return x
