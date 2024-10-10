# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from collections import defaultdict
import torch


class BaseTensorIndex:
    """Indexing for variables in index as Tensor."""

    def __init__(
        self, *, includes: list[str], excludes: list[str], name_to_index: dict[str, int]
    ) -> None:
        """Initialize indexing tensors from includes and excludes using name_to_index.

        Parameters
        ----------
        includes : list[str]
            Variables to include in the indexing that are exclusive to this indexing.
            e.g. Forcing variables for the input indexing, diagnostic variables for the output indexing
        excludes : list[str]
            Variables to exclude from the indexing.
            e.g. Diagnostic variables for the input indexing, forcing variables for the output indexing
        name_to_index : dict[str, int]
            Dictionary mapping variable names to their index in the Tensor.
        """
        self.includes = includes
        self.excludes = excludes
        self.name_to_index = name_to_index

        assert set(self.excludes).issubset(
            self.name_to_index.keys(),
        ), f"Data indexing has invalid entries {[var for var in self.excludes if var not in self.name_to_index]}, not in dataset."
        assert set(self.includes).issubset(
            self.name_to_index.keys(),
        ), f"Data indexing has invalid entries {[var for var in self.includes if var not in self.name_to_index]}, not in dataset."

        self.full = self._build_idx_from_excludes()
        self._only = self._build_idx_from_includes()
        self._removed = self._build_idx_from_includes(self.excludes)
        self.prognostic = self._build_idx_prognostic()
        self.diagnostic = NotImplementedError
        self.forcing = NotImplementedError

    def __len__(self) -> int:
        return len(self.full)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(includes={self.includes}, excludes={self.excludes}, name_to_index={self.name_to_index})"

    def __eq__(self, other):
        if not isinstance(other, BaseTensorIndex):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return (
            torch.allclose(self.full, other.full)
            and torch.allclose(self._only, other._only)
            and torch.allclose(self._removed, other._removed)
            and torch.allclose(self.prognostic, other.prognostic)
            and torch.allclose(self.diagnostic, other.diagnostic)
            and torch.allclose(self.forcing, other.forcing)
            and self.includes == other.includes
            and self.excludes == other.excludes
        )

    def __getitem__(self, key):
        return getattr(self, key)

    def todict(self):
        return {
            "full": self.full,
            "prognostic": self.prognostic,
            "diagnostic": self.diagnostic,
            "forcing": self.forcing,
        }

    @staticmethod
    def representer(dumper, data):
        return dumper.represent_scalar(f"!{data.__class__.__name__}", repr(data))

    def _build_idx_from_excludes(self, excludes=None) -> "torch.Tensor[int]":
        if excludes is None:
            excludes = self.excludes
        return self._build_idx_from_condition(lambda name: name not in excludes)

    def _build_idx_from_includes(self, includes=None) -> "torch.Tensor[int]":
        if includes is None:
            includes = self.includes
        return self._build_idx_from_condition(lambda name: name in includes)

    def _build_idx_prognostic(self) -> "torch.Tensor[int]":
        return self._build_idx_from_excludes(self.includes + self.excludes)

    def _build_idx_from_condition(self, condition):
        # refactor to use two different classes
        typ = type((list(self.name_to_index.values()))[0])
        print(self.name_to_index, typ)

        func = {
            int: self._build_idx_from_condition_todo_i,
            tuple: self._build_idx_from_condition_todo_dict,
        }[typ]
        return func(condition)

    def _build_idx_from_condition_todo_dict(self, condition):
        idx = defaultdict(list)
        for name, (i, j) in self.name_to_index.items():
            assert isinstance(j, int), j
            if condition(name):
                idx[i].append(j)
        return {k: torch.Tensor(sorted(v)).to(torch.int) for k, v in idx.items()}

    def _build_idx_from_condition_todo_i(self, condition):
        idx = []
        for name, i in self.name_to_index.items():
            assert isinstance(i, int), i
            if condition(name):
                idx.append(i)
        return torch.Tensor(sorted(idx)).to(torch.int)


class InputTensorIndex(BaseTensorIndex):
    """Indexing for input variables."""

    def __init__(
        self, *, includes: list[str], excludes: list[str], name_to_index: dict[str, int]
    ) -> None:
        super().__init__(
            includes=includes, excludes=excludes, name_to_index=name_to_index
        )
        self.forcing = self._only
        self.diagnostic = self._removed


class OutputTensorIndex(BaseTensorIndex):
    """Indexing for output variables."""

    def __init__(
        self, *, includes: list[str], excludes: list[str], name_to_index: dict[str, int]
    ) -> None:
        super().__init__(
            includes=includes, excludes=excludes, name_to_index=name_to_index
        )
        self.forcing = self._removed
        self.diagnostic = self._only
