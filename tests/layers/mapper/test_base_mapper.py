# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import dataclass

import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.models.layers.mapper import BaseMapper


@dataclass
class BaseMapperConfig:
    sub_graph: HeteroData
    sub_graph_edge_attributes: list[str]
    in_channels_src: int = 3
    in_channels_dst: int = 3
    hidden_dim: int = 128
    out_channels_dst: int = 5
    cpu_offload: bool = False
    activation: str = "SiLU"
    trainable_size: int = 6


class TestBaseMapper:
    """Test the BaseMapper class."""

    NUM_EDGES: int = 100
    NUM_SRC_NODES: int = 100
    NUM_DST_NODES: int = 200

    @pytest.fixture
    def mapper_init(self, fake_graph):
        return BaseMapperConfig(
            sub_graph=fake_graph[("src", "to", "dst")],
            sub_graph_edge_attributes=["edge_attr1", "edge_attr2"],
        )

    @pytest.fixture
    def mapper(self, mapper_init):
        return BaseMapper(
            in_channels_src=mapper_init.in_channels_src,
            in_channels_dst=mapper_init.in_channels_dst,
            hidden_dim=mapper_init.hidden_dim,
            out_channels_dst=mapper_init.out_channels_dst,
            cpu_offload=mapper_init.cpu_offload,
            activation=mapper_init.activation,
            sub_graph=mapper_init.sub_graph,
            sub_graph_edge_attributes=mapper_init.sub_graph_edge_attributes,
            trainable_size=mapper_init.trainable_size,
        )

    @pytest.fixture
    def pair_tensor(self, mapper_init):
        return (
            torch.rand(mapper_init.in_channels_src, mapper_init.hidden_dim),
            torch.rand(mapper_init.in_channels_dst, mapper_init.hidden_dim),
        )

    @pytest.fixture
    def fake_graph(self) -> HeteroData:
        """Fake graph."""
        graph = HeteroData()
        graph[("src", "to", "dst")].edge_index = torch.concat(
            [
                torch.randint(0, self.NUM_SRC_NODES, (1, self.NUM_EDGES)),
                torch.randint(0, self.NUM_DST_NODES, (1, self.NUM_EDGES)),
            ],
            axis=0,
        )
        graph[("src", "to", "dst")].edge_attr1 = torch.rand((self.NUM_EDGES, 1))
        graph[("src", "to", "dst")].edge_attr2 = torch.rand((self.NUM_EDGES, 32))
        return graph

    def test_initialization(self, mapper, mapper_init):
        assert isinstance(mapper, BaseMapper)
        assert mapper.in_channels_src == mapper_init.in_channels_src
        assert mapper.in_channels_dst == mapper_init.in_channels_dst
        assert mapper.hidden_dim == mapper_init.hidden_dim
        assert mapper.out_channels_dst == mapper_init.out_channels_dst
        assert mapper.activation == mapper_init.activation

    def test_pre_process(self, mapper, pair_tensor):
        x = pair_tensor
        shard_shapes = [list(x[0].shape), list(x[1].shape)]

        x_src, x_dst, shapes_src, shapes_dst = mapper.pre_process(x, shard_shapes)
        assert torch.equal(x_src, x[0])
        assert torch.equal(x_dst, x[1])
        assert shapes_src == shard_shapes[0]
        assert shapes_dst == shard_shapes[1]

    def test_post_process(self, mapper, pair_tensor):
        x_dst = pair_tensor[1]
        shapes_dst = [list(x_dst.shape)]

        result = mapper.post_process(
            x_dst,
            shapes_dst,
        )
        assert torch.equal(result, x_dst)
