from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import torch
from anemoi.utils import DotDict
from torch import nn

from anemoi.models.data_indices.tensor import InputTensorIndex


class BaseBoundingStrategy(nn.Module, ABC):
    """Abstract base class for bounding strategies.

    This class defines an interface for bounding strategies which are used to apply a specific
    restriction to the predictions of a model.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Applies the bounding strategy to the given variables of the input prediction (x)

    Parameters
    ----------
    x : torch.Tensor
        The tensor containing the predictions that will be bounded.

    Returns
    -------
    torch.Tensor
        A tensor with the bounding strategy applied.
    """

    def __init__(
        self,
        *,
        config: DotDict,
        name_to_index: dict,
    ):
        super().__init__()

        self.config = config
        self.name_to_index = name_to_index
        self.variables = self.config["variables"]
        self.data_index = self._create_index(includes=self.variables)

    def _create_index(self, variables: list[str]) -> InputTensorIndex:
        return InputTensorIndex(includes=variables, excludes=None, name_to_index=self.name_to_index)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class ReluBoundingStrategy(BaseBoundingStrategy):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[..., self.data_index] = torch.nn.functional.relu(x[..., self.data_index])
        return x


class HardtanhBoundingStrategy(BaseBoundingStrategy):
    """Initializes the bounding with specified minimum and maximum values for bounding.

    Parameters
    ----------
    min_val : float
        The minimum value for the HardTanh activation.
    max_val : float
        The maximum value for the HardTanh activation.
    """

    def __init__(self, *, config: DotDict, name_to_index: dict, min_val: float, max_val: float):
        super().__init__(config=config, name_to_index=name_to_index)
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[..., self.data_index] = torch.nn.functional.hardtanh(
            x[..., self.data_index], min_val=self.min_val, max_val=self.max_val
        )
        return x


class FractionBoundingStrategy(HardtanhBoundingStrategy):
    """Initializes the FractionBoundingStrategy with specified parameters.

    Parameters
    ----------
    total_var : str
        A string representing a variable from which a secondary variable is derived. For
        example, in the case of convective precipitation (Cp), total_var = Tp (total precipitation).
    """

    def __init__(self, *, config: DotDict, name_to_index: dict, min_val: float, max_val: float, total_var: str):
        super().__init__(config=config, name_to_index=name_to_index, min_val=min_val, max_val=max_val)
        self.total_variable = self._create_index(includes=[total_var])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the HardTanh bounding strategy to the data_index variables
        x = super().forward(x)
        # Calculate the fraction of the total variable
        x[..., self.data_index] *= x[..., self.total_variable]
        return x
