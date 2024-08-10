# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import uuid

import torch
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from torch_geometric.data import HeteroData

from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.preprocessing import Processors

from typing import Optional

class AnemoiModelInterface(torch.nn.Module):
    """An interface for Anemoi models.

    This class is a wrapper around the Anemoi model that includes pre-processing and post-processing steps.
    It inherits from the PyTorch Module class.

    Attributes
    ----------
    config : DotDict
        Configuration settings for the model.
    id : str
        A unique identifier for the model instance.
    multi_step : bool
        Whether the model uses multi-step input.
    graph_data : HeteroData
        Graph data for the model.
    statistics : dict
        Statistics for the data.
    metadata : dict
        Metadata for the model.
    data_indices : dict
        Indices for the data.
    pre_processors : Processors
        Pre-processing steps to apply to the data before passing it to the model.
    post_processors : Processors
        Post-processing steps to apply to the model's output.
    model : AnemoiModelEncProcDec
        The underlying Anemoi model.
    """

    def __init__(
        self, *, config: DotDict, graph_data: HeteroData, statistics: dict, data_indices: dict, metadata: dict, tendency_statistics: Optional[dict] = None
    ) -> None:
        super().__init__()
        self.config = config
        self.id = str(uuid.uuid4())
        self.multi_step = self.config.training.multistep_input
        self.flag_tendency = self.config.training.flag_tendency
        self.graph_data = graph_data
        self.statistics = statistics
        self.tendency_statistics = tendency_statistics
        self.metadata = metadata
        self.data_indices = data_indices
        self._build_model()

    def _build_model(self) -> None:
        """Builds the model and pre- and post-processors."""
        # Instantiate processors for state
        processors_state = [
            [name, instantiate(processor, statistics=self.statistics, data_indices=self.data_indices)]
            for name, processor in self.config.data.processors.state.items()
        ]

        # Assign the processor list pre- and post-processors
        self.pre_processors_state = Processors(processors_state)
        self.post_processors_state = Processors(processors_state, inverse=True)

        # Instantiate processors for tendency
        self.pre_processors_tendency = None
        self.post_processors_tendency = None
        if self.flag_tendency:
            processors_tendency = [
                [name, instantiate(processor, statistics=self.tendency_statistics, data_indices=self.data_indices)]
                for name, processor in self.config.data.processors.tendency.items()
            ]

            self.pre_processors_tendency = Processors(processors_tendency)
            self.post_processors_tendency = Processors(processors_tendency, inverse=True)


        # Instantiate the model (Can be generalised to other models in the future, here we use AnemoiModelEncProcDec)
        self.model = AnemoiModelEncProcDec(
            config=self.config, data_indices=self.data_indices, graph_data=self.graph_data
        )

        # Use the forward method of the model directly
        self.forward = self.model.forward

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Prediction step for the model.

        Parameters
        ----------
        batch : torch.Tensor
            Input batched data.

        Returns
        -------
        torch.Tensor
            Predicted data.
        """
        
        with torch.no_grad():
            
            assert (
                len(batch.shape) == 4
            ), f"The input tensor has an incorrect shape: expected a 4-dimensional tensor, got {batch.shape}!"

            x = self.pre_processors_state(batch[:, 0 : self.multi_step, ...], in_place=False)

            # Dimensions are
            # batch, timesteps, horizontal space, variables
            x = x[..., None, :]  # add dummy ensemble dimension as 3rd index

            #NOTE: TENDENCY CHANGES HERE
            if not self.flag_tendency:
                y_hat = self(x)
                y_hat = self.post_processors(y_hat, in_place=False)
            else:
                tendency_hat = self(x)
                y_hat = self.add_tendency_to_state(batch[:, self.multi_step, ...] , tendency_hat )

        return y_hat
    
    def add_tendency_to_state(self, state_in, tendency_in):
        #NOTE: TENDENCY CHANGES HERE
        state_new = self.post_processors_tendency(tendency_in, in_place=False, data_index=self.data_indices.data.output.full)
        # diagnostic fields are denormallised as full fields
        # NOTE: TENDENCY ISSUE - Explain this ???
        state_new[..., self.data_indices.model.output.diagnostic] = self.post_processors_state(
            tendency_in[..., self.data_indices.model.output.diagnostic],
            in_place=False,
            data_index=self.data_indices.data.output.diagnostic,
        )
        # add state for prognostic variables
        state_new[..., self.data_indices.model.output.prognostic] += state_in[..., self.data_indices.model.input.prognostic]

        return state_new