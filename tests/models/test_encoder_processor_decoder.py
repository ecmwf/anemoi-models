from pathlib import Path

import pytest
import torch

from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec


@pytest.mark.gpu()
@pytest.mark.parametrize(
    "config",
    [
        [],
        ["data.forcing=[]"],
        ["data.forcing=null"],
        ["data.forcing=[lsm]"],
        ["data.diagnostic=[]"],
        ["data.diagnostic=null"],
        ["data.diagnostic=[msl]"],
        ["data.forcing=null", "data.diagnostic=null"],
        ["data.forcing=[]", "data.diagnostic=[]"],
        ["data.forcing=[lsm]", "data.diagnostic=[]"],
        ["data.forcing=[]", "data.diagnostic=[lsm]"],
        ["data.forcing=[lsm]", "data.diagnostic=[tp]"],
    ],
    indirect=True,
)
def test_graph_msg(config, datamodule) -> None:
    device = torch.device("cuda")
    data_indices = datamodule.data_indices
    config.data.num_features = len(datamodule.ds_train.data.variables)
    graph_data = torch.load(Path(config.hardware.paths.graph, config.hardware.files.graph))
    model = AnemoiModelEncProcDec(
        config=config,
        data_indices=data_indices,
        graph_data=graph_data,
    ).to(device)

    input_data = torch.randn(
        config.dataloader.batch_size.training,
        config.training.multistep_input,
        1,
        40320,
        len(data_indices.model.input.full),
        dtype=torch.float32,
        device=device,
    )
    output = torch.randn([config.dataloader.batch_size.training, 1, 40320, len(data_indices.model.output.full)])
    assert model.forward(input_data).shape == output.shape, "Output shape is not correct"
