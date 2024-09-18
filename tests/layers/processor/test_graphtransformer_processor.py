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

from anemoi.models.layers.graph import TrainableTensor
from anemoi.models.layers.processor import GraphTransformerProcessor


@dataclass
class GraphTransformerProcessorConfig:
    sub_graph: HeteroData
    edge_attributes: list[str]
    num_layers: int = 2
    num_channels: int = 128
    num_chunks: int = 2
    num_heads: int = 16
    mlp_hidden_ratio: int = 4
    activation: str = "GELU"
    cpu_offload: bool = False
    src_grid_size: int = 7
    dst_grid_size: int = 13
    trainable_size: int = 6


class TestGraphTransformerProcessor:
    """Test the GraphTransformerProcessor class."""

    NUM_NODES: int = 100
    NUM_EDGES: int = 200

    @pytest.fixture
    def fake_graph(self) -> tuple[HeteroData, int]:
        graph = HeteroData()
        graph["nodes"].x = torch.rand((self.NUM_NODES, 2))
        graph[("nodes", "to", "nodes")].edge_index = torch.randint(0, self.NUM_NODES, (2, self.NUM_EDGES))
        graph[("nodes", "to", "nodes")].edge_attr1 = torch.rand((self.NUM_EDGES, 3))
        graph[("nodes", "to", "nodes")].edge_attr2 = torch.rand((self.NUM_EDGES, 4))
        return graph

    @pytest.fixture
    def graphtransformer_init(self, fake_graph: HeteroData):
        return GraphTransformerProcessorConfig(
            sub_graph=fake_graph[("nodes", "to", "nodes")], edge_attributes=["edge_attr1", "edge_attr2"]
        )

    @pytest.fixture
    def graphtransformer_processor(self, graphtransformer_init):
        return GraphTransformerProcessor(
            num_layers=graphtransformer_init.num_layers,
            num_channels=graphtransformer_init.num_channels,
            num_chunks=graphtransformer_init.num_chunks,
            num_heads=graphtransformer_init.num_heads,
            mlp_hidden_ratio=graphtransformer_init.mlp_hidden_ratio,
            activation=graphtransformer_init.activation,
            cpu_offload=graphtransformer_init.cpu_offload,
            sub_graph=graphtransformer_init.sub_graph,
            sub_graph_edge_attributes=graphtransformer_init.edge_attributes,
            src_grid_size=graphtransformer_init.src_grid_size,
            dst_grid_size=graphtransformer_init.dst_grid_size,
            trainable_size=graphtransformer_init.trainable_size,
        )

    def test_graphtransformer_processor_init(self, graphtransformer_processor, graphtransformer_init):
        num_layers = graphtransformer_init.num_layers
        num_channels = graphtransformer_init.num_channels
        num_chunks = graphtransformer_init.num_chunks
        assert graphtransformer_processor.num_chunks == num_chunks
        assert graphtransformer_processor.num_channels == num_channels
        assert graphtransformer_processor.chunk_size == num_layers // num_chunks
        assert isinstance(graphtransformer_processor.trainable, TrainableTensor)

    def test_forward(self, graphtransformer_processor, graphtransformer_init):
        batch_size = 1
        num_channels = graphtransformer_init.num_channels
        trainable_size = graphtransformer_init.trainable_size
        x = torch.rand((self.NUM_EDGES, num_channels))
        shard_shapes = [list(x.shape)]

        # Run forward pass of processor
        output = graphtransformer_processor.forward(x, batch_size, shard_shapes)
        assert output.shape == (self.NUM_EDGES, num_channels)

        # Generate dummy target and loss function
        loss_fn = torch.nn.MSELoss()
        target = torch.rand((self.NUM_EDGES, num_channels))
        loss = loss_fn(output, target)

        # Check loss
        assert loss.item() >= 0

        # Backward pass
        loss.backward()

        # Check gradients of trainable tensor
        assert graphtransformer_processor.trainable.trainable.grad.shape == (
            self.NUM_EDGES,
            trainable_size,
        )

        # Check gradients of processor
        for param in graphtransformer_processor.parameters():
            assert param.grad is not None, f"param.grad is None for {param}"
            assert (
                param.grad.shape == param.shape
            ), f"param.grad.shape ({param.grad.shape}) != param.shape ({param.shape}) for {param}"
