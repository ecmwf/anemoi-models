# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data.data_indices.collection import IndexCollection
from anemoi.models.data.normalizer import InputNormalizer


@pytest.fixture()
def input_normalizer():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "normalizer": {"default": "mean-std", "min-max": ["x"], "max": ["y"], "none": ["z"], "mean-std": ["q"]},
                "forcing": ["z", "q"],
                "diagnostic": ["other"],
            },
        },
    )
    statistics = {
        "mean": np.array([1.0, 2.0, 3.0, 4.5, 3.0]),
        "stdev": np.array([0.5, 0.5, 0.5, 1, 14]),
        "minimum": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "maximum": np.array([11.0, 10.0, 10.0, 10.0, 10.0]),
    }
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "other": 4}
    # data_indices = {"dynamic": torch.Tensor([0, 1, 4]).to(torch.int), "output": torch.Tensor([0, 1, 2, 3, 4]).to(torch.int)}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    return InputNormalizer(config=config, statistics=statistics, data_indices=data_indices)


def test_normalizer_not_inplace(input_normalizer) -> None:
    x = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
    input_normalizer(x, in_place=False)
    assert torch.allclose(x, torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]))


def test_normalizer_inplace(input_normalizer) -> None:
    x = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
    out = input_normalizer(x, in_place=True)
    assert not torch.allclose(x, torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]))
    assert torch.allclose(x, out)


def test_normalize(input_normalizer) -> None:
    x = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
    expected_output = torch.Tensor([[0.0, 0.2, 3.0, -0.5, 1 / 7], [0.5, 0.7, 8.0, 4.5, 0.5]])
    assert torch.allclose(input_normalizer.normalize(x), expected_output)


def test_normalize_small(input_normalizer) -> None:
    x = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
    expected_output = torch.Tensor([[0.0, 0.2, 3.0, -0.5], [0.5, 0.7, 8.0, 4.5]])
    assert torch.allclose(
        input_normalizer.normalize(x[..., [0, 1, 2, 3]], data_index=[0, 1, 2, 3], in_place=False),
        expected_output,
    )
    assert torch.allclose(input_normalizer.normalize(x[..., [0, 1, 2, 3]]), expected_output)


def test_denormalize_small(input_normalizer) -> None:
    expected_output = torch.Tensor([[1.0, 2.0, 5.0], [6.0, 7.0, 10.0]])
    x = torch.Tensor([[0.0, 0.2, 1 / 7], [0.5, 0.7, 0.5]])
    assert torch.allclose(input_normalizer.denormalize(x, data_index=[0, 1, 4], in_place=False), expected_output)
    assert torch.allclose(input_normalizer.denormalize(x), expected_output)


def test_denormalize(input_normalizer) -> None:
    x = torch.Tensor([[0.0, 0.2, 3.0, -0.5, 1 / 7], [0.5, 0.7, 8.0, 4.5, 0.5]])
    expected_output = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
    assert torch.allclose(input_normalizer.denormalize(x), expected_output)


def test_normalize_denormalize(input_normalizer) -> None:
    x = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
    assert torch.allclose(
        input_normalizer.denormalize(input_normalizer.normalize(x, in_place=False), in_place=False), x
    )
