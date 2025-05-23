"""Small utility functions"""
import os
import platform
import re
from typing import Tuple

import psutil
import torch


def n_workers() -> int:
    """
    Get the number of workers to use for data loading.

    This is the maximum number of CPUs allowed for the process, scaled for the
    number of GPUs being used.

    On Windows and MacOS, we only use the main process. See:
    https://discuss.pytorch.org/t/errors-when-using-num-workers-0-in-dataloader/97564/4
    https://github.com/pytorch/pytorch/issues/70344

    Returns
    -------
    int
        The number of workers.
    """
    # Windows or MacOS: no multiprocessing.
    if platform.system() in ["Windows", "Darwin"]:
        return 0
    # Linux: scale the number of workers by the number of GPUs (if present).
    try:
        n_cpu = len(psutil.Process().cpu_affinity())
    except AttributeError:
        n_cpu = os.cpu_count()
    return (
        n_cpu // n_gpu if (n_gpu := torch.cuda.device_count()) > 1 else n_cpu
    )


def split_version(version: str) -> Tuple[str, str, str]:
    """
    Split the version into its semantic versioning components.

    Parameters
    ----------
    version : str
        The version number.

    Returns
    -------
    major : str
        The major release.
    minor : str
        The minor release.
    patch : str
        The patch release.
    """
    version_regex = re.compile(r"(\d+)\.(\d+)\.*(\d*)(?:.dev\d+.+)?")
    return tuple(g for g in version_regex.match(version).groups())


import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)


def read_tensorboard_scalars(path):
    """Read scalars from Tensorboard logs.

    Parameters
    ----------
    path : str
        The path of the scalar log file.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing the scalar values.
    """
    event = EventAccumulator(path)
    event.Reload()
    data = []
    for tag in event.Tags()["scalars"]:
        tag_df = pd.DataFrame(
            event.Scalars(tag), columns=["wall_time", "step", "value"]
        )
        tag_df["tag"] = tag
        data.append(tag_df)

    return pd.concat(data)


def listify(obj):
    """Turn an object into a list, but don't split strings."""
    try:
        assert not isinstance(obj, str)
        iter(obj)
    except (AssertionError, TypeError):
        obj = [obj]

    return list(obj)


# For Parameter Checking ------------------------------------------------------
def check_int(integer, name):
    """Verify that an object is an integer, or coercible to one.

    Parameters
    ----------
    integer : int
        The integer to check.
    name : str
        The name to print in the error message if it fails.
    """
    if isinstance(integer, int):
        return integer

    # Else if it is a float:
    coerced = int(integer)
    if coerced != integer:
        raise ValueError(f"'{name}' must be an integer.")

    return coerced


def check_positive_int(integer, name):
    """Verify that an object is an integer and positive.

    Parameters
    ----------
    integer : int
        The integer to check.
    name : str
        The name to print in the error message if it fails.
    """
    try:
        integer = check_int(integer, name)
        assert integer > 0
    except (ValueError, AssertionError):
        raise ValueError(f"'{name}' must be a positive integer.")

    return integer
