# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import dataclass

import hypothesis.strategies as st
import torch
from hypothesis import given
from hypothesis import settings

from anemoi.models.layers.mlp import MLP


@dataclass
class MLPConfig:
    in_features: int = 48
    hidden_dim: int = 23
    out_features: int = 27
    n_extra_layers: int = 0
    activation: str = "SiLU"
    final_activation: bool = False
    layer_norm: bool = False
    checkpoints: bool = False


from_config = dict(
    init_config=st.builds(
        MLPConfig,
        in_features=st.integers(min_value=1, max_value=100),
        hidden_dim=st.integers(min_value=1, max_value=100),
        out_features=st.integers(min_value=1, max_value=100),
        n_extra_layers=st.integers(min_value=0, max_value=10),
        activation=st.sampled_from(("ReLU", "SiLU", "GELU")),
        final_activation=st.booleans(),
        layer_norm=st.booleans(),
        checkpoints=st.booleans(),
    )
)

run_model = dict(
    batch_size=st.integers(min_value=1, max_value=2),
    num_gridpoints=st.integers(min_value=1, max_value=512),
)


class TestMLP:

    def create_model(self, init_config):
        return MLP(
            init_config.in_features,
            init_config.hidden_dim,
            init_config.out_features,
            init_config.n_extra_layers,
            init_config.activation,
            init_config.final_activation,
            init_config.layer_norm,
            init_config.checkpoints,
        )

    @given(**from_config)
    def test_init(self, init_config):
        """Test MLP initialization."""
        mlp = self.create_model(init_config)

        assert isinstance(mlp, MLP)
        if isinstance(mlp.model, torch.nn.Sequential):
            length = 3 + 2 * (init_config.n_extra_layers + 1) + init_config.layer_norm + init_config.final_activation
            assert len(mlp.model) == length

    @settings(deadline=None)
    @given(**run_model, **from_config)
    def test_forwards(self, batch_size, num_gridpoints, init_config):
        """Test MLP forward pass."""
        mlp = self.create_model(init_config)
        num_features = init_config.in_features
        num_out_feature = init_config.out_features
        x_in = torch.randn((batch_size, num_gridpoints, num_features), dtype=torch.float32, requires_grad=True)

        out = mlp(x_in)
        assert out.shape == (
            batch_size,
            num_gridpoints,
            num_out_feature,
        ), "Output shape is not correct"

    @given(**run_model, **from_config)
    def test_backward(self, batch_size, num_gridpoints, init_config):
        """Test MLP backward pass."""
        mlp = self.create_model(init_config)
        num_features = init_config.in_features
        num_out_feature = init_config.out_features

        x_in = torch.randn((batch_size, num_gridpoints, num_features), dtype=torch.float32, requires_grad=True)

        y = mlp(x_in)
        assert y.shape == (batch_size, num_gridpoints, num_out_feature)

        loss = y.sum()
        print("running backward on the dummy loss ...")
        loss.backward()

        for param in mlp.parameters():
            assert param.grad is not None, f"param.grad is None for {param}"
            assert (
                param.grad.shape == param.shape
            ), f"param.grad.shape ({param.grad.shape}) != param.shape ({param.shape}) for {param}"
