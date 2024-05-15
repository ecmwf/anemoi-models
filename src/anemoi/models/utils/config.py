# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


class DotConfig(dict):
    """A Config dictionary that allows access to its keys as attributes."""

    def __init__(self, *args, **kwargs) -> None:
        for a in args:
            self.update(a)
        self.update(kwargs)

    def __getattr__(self, name):
        if name in self:
            x = self[name]
            if isinstance(x, dict):
                return DotConfig(x)
            return x
        raise AttributeError(name)
