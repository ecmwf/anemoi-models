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
from torch import nn

LOGGER = logging.getLogger(__name__)


class InputNormalizer(nn.Module):
    """Normalizes input data to zero mean and unit variance."""

    def __init__(self, *, config, statistics: dict, data_indices: dict) -> None:
        """Initialize the normalizer.

        Parameters
        ----------
        zarr_metadata : Dict
            Zarr metadata dictionary
        """
        super().__init__()
        LOGGER.setLevel(config.diagnostics.log.code.level)

        default = config.data.normalizer.default
        method_config = {k: v for k, v in config.data.normalizer.items() if k != "default" and v is not None}

        if not method_config:
            LOGGER.warning(
                f"Normalizing: Using default method {default} for all variables not specified in the config."
            )

        name_to_index = data_indices.data.input.name_to_index

        methods = {
            variable: method
            for method, variables in method_config.items()
            if not isinstance(variables, str)
            for variable in variables
        }

        assert len(methods) == sum(len(v) for v in method_config.values()), (
            f"Error parsing methods in InputNormalizer methods ({len(methods)}) "
            f"and entries in config ({sum(len(v) for v in method_config)}) do not match."
        )

        minimum = statistics["minimum"]
        maximum = statistics["maximum"]
        mean = statistics["mean"]
        stdev = statistics["stdev"]

        n = minimum.size
        assert maximum.size == n, (maximum.size, n)
        assert mean.size == n, (mean.size, n)
        assert stdev.size == n, (stdev.size, n)

        assert isinstance(methods, dict)
        for name, method in methods.items():
            assert name in name_to_index, f"{name} is not a valid variable name"
            assert method in [
                "mean-std",
                # "robust",
                "min-max",
                "max",
                "none",
            ], f"{method} is not a valid normalisation method"

        _norm_add = np.zeros((n,), dtype=np.float32)
        _norm_mul = np.ones((n,), dtype=np.float32)

        for name, i in name_to_index.items():
            m = methods.get(name, default)
            if m == "mean-std":
                LOGGER.debug(f"Normalizing: {name} is mean-std-normalised.")
                if stdev[i] < (mean[i] * 1e-6):
                    warnings.warn(f"Normalizing: the field seems to have only one value {mean[i]}")
                _norm_mul[i] = 1 / stdev[i]
                _norm_add[i] = -mean[i] / stdev[i]

            elif m == "min-max":
                LOGGER.debug(f"Normalizing: {name} is min-max-normalised to [0, 1].")
                x = maximum[i] - minimum[i]
                if x < 1e-9:
                    warnings.warn(f"Normalizing: the field {name} seems to have only one value {maximum[i]}.")
                _norm_mul[i] = 1 / x
                _norm_add[i] = -minimum[i] / x

            elif m == "max":
                LOGGER.debug(f"Normalizing: {name} is max-normalised to [0, 1].")
                _norm_mul[i] = 1 / maximum[i]

            elif m == "none":
                LOGGER.info(f"Normalizing: {name} is not normalized.")

            else:
                raise ValueError[f"Unknown normalisation method for {name}: {m}"]

        # register buffer - this will ensure they get copied to the correct device(s)
        self.register_buffer("_norm_mul", torch.from_numpy(_norm_mul), persistent=True)
        self.register_buffer("_norm_add", torch.from_numpy(_norm_add), persistent=True)
        self.register_buffer("_input_idx", data_indices.data.input.full, persistent=True)
        self.register_buffer("_output_idx", data_indices.data.output.full, persistent=True)

    def normalize(
        self, x: torch.Tensor, in_place: bool = True, data_index: Optional[torch.Tensor] = None
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

        if data_index is not None:
            x[..., :] = x[..., :] * self._norm_mul[data_index] + self._norm_add[data_index]
        elif x.shape[-1] == len(self._input_idx):
            x[..., :] = x[..., :] * self._norm_mul[self._input_idx] + self._norm_add[self._input_idx]
        else:
            x[..., :] = x[..., :] * self._norm_mul + self._norm_add
        return x

    def forward(self, x: torch.Tensor, in_place: bool = True) -> torch.Tensor:
        return self.normalize(x, in_place=in_place)

    def denormalize(
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

        # Denormalize dynamic or full tensors
        # input and predicted tensors have different shapes
        # hence, we mask out the forcing indices
        if data_index is not None:
            x[..., :] = (x[..., :] - self._norm_add[data_index]) / self._norm_mul[data_index]
        elif x.shape[-1] == len(self._output_idx):
            x[..., :] = (x[..., :] - self._norm_add[self._output_idx]) / self._norm_mul[self._output_idx]
        else:
            x[..., :] = (x[..., :] - self._norm_add) / self._norm_mul
        return x
