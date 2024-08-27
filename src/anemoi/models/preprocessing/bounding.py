from abc import ABC
from abc import abstractmethod

import torch
from torch import nn


class BaseBoundingStrategy(nn.Module, ABC):
    """Abstract base class for bounding strategies.

    This class defines an interface for bounding strategies which are used to apply a specific
    restriction to the predictions of a model.

    Methods
    -------
    forward(y_pred: torch.Tensor, indices: list) -> torch.Tensor
        Applies the bounding strategy to the given variables (indices) of the input prediction (y_pred)

    Parameters
    ----------
    y_pred : torch.Tensor
        The tensor containing the predictions that will be bounded.
    indices : list
        A list of indices specifying which variables in `y_pred` should be bounded.

    Returns
    -------
    torch.Tensor
        A tensor with the bounding strategy applied.
    """

    @abstractmethod
    def forward(self, y_pred: torch.Tensor, indices: list) -> torch.Tensor:
        pass


class ReluBoundingStrategy(BaseBoundingStrategy):
    def forward(self, y_pred: torch.Tensor, indices: list) -> torch.Tensor:
        return torch.nn.functional.relu(y_pred[..., indices[0]])


class HardtanhBoundingStrategy(BaseBoundingStrategy):
    """Initializes the bounding with specified minimum and maximum values for bounding.

    Parameters
    ----------
    min_val : float
        The minimum value for the HardTanh activation.
    max_val : float
        The maximum value for the HardTanh activation.
    """

    def __init__(self, min_val: float, max_val: float):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, y_pred: torch.Tensor, indices: list) -> torch.Tensor:
        return torch.nn.functional.hardtanh(y_pred[..., indices[0]], min_val=self.min_val, max_val=self.max_val)


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

    def __init__(self, min_val: float, max_val: float, total_var: str):

        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.total_var = total_var

    def forward(self, y_pred: torch.Tensor, indices: list) -> torch.Tensor:
        return (
            torch.nn.functional.hardtanh(y_pred[..., indices[0]], min_val=self.min_val, max_val=self.max_val)
            * y_pred[..., indices[1]]
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

    def __init__(self, min_val: float, max_val: float, first_var: str, second_var: str):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.first_var = first_var
        self.second_var = second_var

    def forward(self, y_pred: torch.Tensor, indices: list) -> torch.Tensor:
        return torch.nn.functional.hardtanh(y_pred[..., indices[0]], min_val=self.min_val, max_val=self.max_val) * (
            y_pred[..., indices[1]] - y_pred[..., indices[2]]
        )
