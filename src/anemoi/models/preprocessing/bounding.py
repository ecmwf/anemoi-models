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
    forward(x: torch.Tensor, indices: list) -> torch.Tensor
        Applies the bounding strategy to the given variables (indices) of the input prediction (x)

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
        return torch.nn.functional.relu(x[..., self.data_index])


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
        return torch.nn.functional.hardtanh(x[..., self.data_index], min_val=self.min_val, max_val=self.max_val)


class FractionHardtanhBoundingStrategy(BaseBoundingStrategy):
    """Initializes the FractionHardtanhBoundingStrategy with specified parameters.

    Parameters
    ----------
    min_val : float
        The minimum value for the HardTanh activation function.
    max_val : float
        The maximum value for the HardTanh activation function.
    total_var : str
        A string representing a variable from which a secondary variable is derived. For
        example, in the case of convective precipitation (Cp), total_var = Tp (total precipitation).
    """

    def __init__(self, *, config: DotDict, name_to_index: dict, min_val: float, max_val: float, total_var: str):
        super().__init__(config=config, name_to_index=name_to_index)
        self.min_val = min_val
        self.max_val = max_val
        self.total_variable = self._create_index(includes=[total_var])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            torch.nn.functional.hardtanh(x[..., self.data_index], min_val=self.min_val, max_val=self.max_val)
            * x[..., self.total_variable]
        )


class CustomFractionHardtanhBoundingStrategy(BaseBoundingStrategy):
    """Initializes the CustomFractionHardtanhBoundingStrategy.

    Description
    ----------
    Initializes the CustomFractionHardtanhBoundingStrategy with specified
    parameters. This is a special case of FractionHardtanhBoundingStrategy where the
    total variable is constructed from a combination of two other variables. For
    example, large-scale precipitation (lsp) can be derived from total precipitation (tp)
    and convective precipitation (cp) as follows: lsp = tp - cp.

    Parameters
    ----------
    min_val : float
        The minimum value for the HardTanh activation function.
    max_val : float
        The maximum value for the HardTanh activation function.
    first_var : str
        First variable from which the total variable is derived.
    second_var : str
        Second variable from which the total variable is derived.
    """

    def __init__(
        self, *, config: DotDict, name_to_index: dict, min_val: float, max_val: float, first_var: str, second_var: str
    ):
        super().__init__(config=config, name_to_index=name_to_index)
        self.min_val = min_val
        self.max_val = max_val
        self.first_var = self._create_index(includes=[first_var])
        self.second_var = self._create_index(includes=[second_var])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.hardtanh(x[..., self.data_index], min_val=self.min_val, max_val=self.max_val) * (
            x[..., self.first_var] - x[..., self.second_var]
        )
