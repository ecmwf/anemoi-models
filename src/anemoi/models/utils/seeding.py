# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import os


def get_base_seed(env_var_list=("AIFS_BASE_SEED", "SLURM_JOB_ID")) -> int:
    """Gets the base seed from the environment variables.

    Option to manually set a seed via export AIFS_BASE_SEED=xxx in job script
    """
    base_seed = None
    for env_var in env_var_list:
        if env_var in os.environ:
            base_seed = int(os.environ.get(env_var))
            break

    assert base_seed is not None, f"Base seed not found in environment variables {env_var_list}"

    if base_seed < 1000:
        base_seed = base_seed * 1000  # make it (hopefully) big enough

    return base_seed
