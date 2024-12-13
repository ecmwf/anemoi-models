# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import ABC
from typing import Optional

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import BasePreprocessor
from anemoi.models.preprocessing.monomapper import Monomapper
from anemoi.models.preprocessing.multimapper import Multimapper

LOGGER = logging.getLogger(__name__)


class Remapper(BasePreprocessor, ABC):
    """Remap and convert variables for single variables."""

    def __new__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        super().__init__(config, data_indices, statistics)
        monomappings = Monomapper.supported_methods
        multimappings = Multimapper.supported_methods
        if self.method in monomappings:
            return Monomapper(config, data_indices, statistics)
        elif self.add_module in multimappings:
            return Multimapper(config, data_indices, statistics)
        else:
            raise ValueError(f"Unknown remapping method: {self.method}")
