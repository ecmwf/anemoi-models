# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from anemoi.models.data.data_indices.collection import IndexCollection
from anemoi.models.data.data_indices.index import BaseIndex
from anemoi.models.data.data_indices.tensor import BaseTensorIndex


@pytest.mark.data_dependent()
def test_datamodule_datasets(datamodule) -> None:
    assert hasattr(datamodule, "dataset_train")
    assert hasattr(datamodule, "dataset_valid")
    assert hasattr(datamodule, "dataset_test")


def test_datamodule_dataloaders(datamodule) -> None:
    assert hasattr(datamodule, "train_dataloader")
    assert hasattr(datamodule, "val_dataloader")
    assert hasattr(datamodule, "test_dataloader")


@pytest.mark.data_dependent()
def test_datamodule_metadata(datamodule) -> None:
    assert hasattr(datamodule, "metadata")
    assert isinstance(datamodule.metadata, dict)


@pytest.mark.data_dependent()
def test_datamodule_statistics(datamodule) -> None:
    assert hasattr(datamodule, "statistics")
    assert isinstance(datamodule.statistics, dict)
    assert "mean" in datamodule.statistics
    assert "stdev" in datamodule.statistics
    assert "minimum" in datamodule.statistics
    assert "maximum" in datamodule.statistics


@pytest.mark.data_dependent()
@pytest.mark.parametrize(
    ("data_model", "in_out", "full_only_prognostic"),
    [
        (a, b, c)
        for a in ["data", "model"]
        for b in ["input", "output"]
        for c in ["full", "forcing", "diagnostic", "prognostic"]
    ],
)
def test_datamodule_api(datamodule, data_model, in_out, full_only_prognostic) -> None:
    assert hasattr(datamodule, "data_indices")
    assert isinstance(datamodule.data_indices, IndexCollection)
    assert hasattr(datamodule.data_indices, data_model)
    assert isinstance(datamodule.data_indices[data_model], BaseIndex)
    data_indices = getattr(datamodule.data_indices, data_model)
    assert isinstance(getattr(data_indices, in_out), BaseTensorIndex)
    assert hasattr(getattr(data_indices, in_out), full_only_prognostic)
    assert isinstance(getattr(getattr(data_indices, in_out), full_only_prognostic), torch.Tensor)


@pytest.mark.data_dependent()
def test_datamodule_data_indices(datamodule) -> None:
    # Check that different indices are split correctly
    all_data = set(datamodule.data_indices.data.input.name_to_index.values())
    assert (
        set(datamodule.data_indices.data.input.full.numpy()).union(
            datamodule.data_indices.data.input.name_to_index[v] for v in datamodule.config.data.diagnostic
        )
        == all_data
    )
    assert len(datamodule.data_indices.data.input.prognostic) <= len(datamodule.data_indices.data.input.full)
    assert len(datamodule.data_indices.data.output.prognostic) <= len(datamodule.data_indices.data.output.full)
    assert len(datamodule.data_indices.data.output.prognostic) == len(datamodule.data_indices.data.input.prognostic)

    assert len(datamodule.data_indices.model.input.prognostic) <= len(datamodule.data_indices.model.input.full)
    assert len(datamodule.data_indices.model.output.prognostic) <= len(datamodule.data_indices.model.output.full)
    assert len(datamodule.data_indices.model.output.prognostic) == len(datamodule.data_indices.model.input.prognostic)


@pytest.mark.data_dependent()
def test_datamodule_batch(datamodule) -> None:
    first_batch = next(iter(datamodule.train_dataloader()))
    assert isinstance(first_batch, torch.Tensor)
    assert first_batch.shape[-1] == len(
        datamodule.data_indices.data.input.name_to_index.values(),
    ), "Batch should have all variables"
    assert (
        first_batch.shape[0] == datamodule.config.dataloader.batch_size.training
    ), "Batch should have correct batch size"
    assert (
        first_batch.shape[1] == datamodule.config.training.multistep_input + 1
    ), "Batch needs correct sequence length (steps + 1)"
