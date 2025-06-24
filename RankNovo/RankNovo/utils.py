"""Small utility functions"""
import os
import platform
import re
from typing import Tuple

import psutil
import torch


def n_workers() -> int:
    """

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
