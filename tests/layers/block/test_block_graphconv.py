# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from anemoi.models.layers.block import MLP
from anemoi.models.layers.block import GraphConvMapperBlock
from anemoi.models.layers.block import GraphConvProcessorBlock
from anemoi.models.layers.conv import GraphConv


class TestGraphConvProcessorBlock:
    @given(
        in_channels=st.integers(min_value=1, max_value=100),
        out_channels=st.integers(min_value=1, max_value=100),
        mlp_extra_layers=st.integers(min_value=1, max_value=5),
        activation=st.sampled_from(["ReLU", "GELU", "Tanh"]),
        update_src_nodes=st.booleans(),
        num_chunks=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=10)
    def test_init(
        self,
        in_channels,
        out_channels,
        mlp_extra_layers,
        activation,
        update_src_nodes,
        num_chunks,
    ):
        block = GraphConvProcessorBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            mlp_extra_layers=mlp_extra_layers,
            activation=activation,
            update_src_nodes=update_src_nodes,
            num_chunks=num_chunks,
        )

        assert isinstance(block, GraphConvProcessorBlock)
        assert isinstance(block.node_mlp, MLP)
        assert isinstance(block.conv, GraphConv)

        assert block.update_src_nodes == update_src_nodes
        assert block.num_chunks == num_chunks


class TestGraphConvMapperBlock:
    @given(
        in_channels=st.integers(min_value=1, max_value=100),
        out_channels=st.integers(min_value=1, max_value=100),
        mlp_extra_layers=st.integers(min_value=1, max_value=5),
        activation=st.sampled_from(["ReLU", "GELU", "Tanh"]),
        update_src_nodes=st.booleans(),
        num_chunks=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=10)
    def test_init(
        self,
        in_channels,
        out_channels,
        mlp_extra_layers,
        activation,
        update_src_nodes,
        num_chunks,
    ):
        block = GraphConvMapperBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            mlp_extra_layers=mlp_extra_layers,
            activation=activation,
            update_src_nodes=update_src_nodes,
            num_chunks=num_chunks,
        )

        assert isinstance(block, GraphConvMapperBlock)
        assert isinstance(block.node_mlp, MLP)
        assert isinstance(block.conv, GraphConv)

        assert block.update_src_nodes == update_src_nodes
        assert block.num_chunks == num_chunks
