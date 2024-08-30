# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing.log_normalizer import InputLogNormalizer


@pytest.fixture()
def input_log_normalizer():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "log_normalizer": {"log": ["x"], "log-piecewise": ["other"]},
                #"log_normalizer": {"log": ["x"],},
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
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    print(data_indices.data.input.name_to_index)
    return InputLogNormalizer(config=config.data.log_normalizer, statistics=statistics, data_indices=data_indices)


def test_log_normalizer_not_inplace(input_log_normalizer) -> None:
    x = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
    input_log_normalizer(x, in_place=False)
    assert torch.allclose(x, torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]))


def test_log_normalizer_inplace(input_log_normalizer) -> None:
    x = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
    out = input_log_normalizer(x, in_place=True)
    assert not torch.allclose(x, torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]))
    assert torch.allclose(x, out)


def test_log_normalize(input_log_normalizer) -> None:
    x = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 0.5], [6.0, 7.0, 8.0, 9.0, 10.0]])
    expected_output = torch.Tensor([[0.6931471805599453, 2.0, 3.0, 4.0, 0.5],  # log(1+1) = 0.6931471805599453
                                    [1.9459101490553132, 7.0, 8.0, 9.0, 2.0]]) 
    assert torch.allclose(input_log_normalizer.transform(x), expected_output)


def test_inverse_log_transform(input_log_normalizer) -> None:
    x = torch.Tensor([[0.6931471805599453, 2.0, 3.0, 4.0, 0.5],  # log(1+1) = 0.6931471805599453
                      [1.9459101490553132, 7.0, 8.0, 9.0, 2.0]])
    expected_output = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 0.5], [6.0, 7.0, 8.0, 9.0, 10.0]])
    assert torch.allclose(input_log_normalizer.inverse_transform(x), expected_output)
