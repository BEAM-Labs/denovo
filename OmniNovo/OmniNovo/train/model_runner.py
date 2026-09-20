"""Inference orchestration for the de novo sequencing model."""
import glob
import os
import tempfile
import uuid
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies import DDPStrategy
from loguru import logger

from .. import utils
from .db_dataloader import DeNovoDataModule
from .model import Spec2Pep


def predict(
    peak_path: str,
    model_filename: str,
    config: Dict[str, Any],
    out_writer=None,
    log_base: str = None,
) -> None:
    """Predict peptide sequences with a supplied model checkpoint.

    Parameters
    ----------
    peak_path : str
        The path with peak files for predicting peptide sequences.
    model_filename : str
        The file name of the model weights (.ckpt file).
    config : Dict[str, Any]
        The configuration options.
    out_writer : optional
        The output writer to export prediction results.
    log_base : str, optional
        Directory for prediction TSV output.
    """
    _execute_existing(peak_path, model_filename, config, out_writer=out_writer, log_base=log_base)


def _execute_existing(
    peak_path: str,
    model_filename: str,
    config: Dict[str, Any],
    out_writer=None,
    log_base=None,
) -> None:
    """Run inference on the supplied spectra.

    Parameters
    ----------
    peak_path : str
        The path with peak files for predicting peptide sequences.
    model_filename : str
        The file name of the model weights (.ckpt file).
    config : Dict[str, Any]
        The configuration options.
    out_writer : optional
        The output writer to export prediction results.
    log_base : str, optional
        Directory for prediction TSV output.
    """
    # Load the supplied model checkpoint.
    if not os.path.isfile(model_filename):
        logger.error(f"Could not find the model weights at file {model_filename}")
        raise FileNotFoundError("Could not find the model weights")

    if log_base is None or len(log_base) < 5:
        log_filename = None
    else:
        if not os.path.exists(log_base):
            os.makedirs(log_base, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(peak_path))[0] + ".json"
        log_filename = os.path.join(log_base, base_name)
        if os.path.exists(log_filename):
            os.remove(log_filename)

    model = Spec2Pep(
        residues=config["residues"],
        legal_ptms=config.get("legal_ptms"),
        num_key_value_heads=config.get("num_key_value_heads"),
        head_dim=config.get("head_dim"),
        flash_attn=config.get("flash_attn"),
        use_bf16=config.get("use_bf16"),
        rules=config.get("rules"),
        output_site_probs=config.get("output_site_probs", False),
        log_filename=log_filename,
    ).load_from_checkpoint(
        model_filename,
        PMC_enable=config["PMC_enable"],
        mass_control_tol=config["mass_control_tol"],
        dim_model=config["dim_model"],
        n_head=config["n_head"],
        dim_feedforward=config["dim_feedforward"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        dim_intensity=config["dim_intensity"],
        custom_encoder=config["custom_encoder"],
        max_length=config["max_length"],
        residues=config["residues"],
        max_charge=config["max_charge"],
        precursor_mass_tol=config["precursor_mass_tol"],
        isotope_error_range=config["isotope_error_range"],
        n_beams=config["n_beams"],
        n_log=config["n_log"],
        out_writer=out_writer,
        legal_ptms=config.get("legal_ptms"),
        num_key_value_heads=config.get("num_key_value_heads"),
        head_dim=config.get("head_dim"),
        flash_attn=config.get("flash_attn"),
        use_bf16=config.get("use_bf16"),
        rules=config.get("rules"),
        output_site_probs=config.get("output_site_probs", False),
        log_filename=log_filename,
    )

    # Read the MS/MS spectra for inference.
    peak_ext = (".mgf", ".mzml", ".mzxml", ".h5", ".hdf5")
    if len(peak_filenames := _get_peak_filenames(peak_path, peak_ext)) == 0:
        logger.error(f"Could not find peak files from {peak_path}")
        raise FileNotFoundError("Could not find peak files")

    peak_is_not_index = any(
        os.path.splitext(fn)[1] in (".mgf", ".mzxml", ".mzml") for fn in peak_filenames
    )

    # Use configurable temp directory
    temp_dir = config.get("temp_dir", tempfile.gettempdir())

    class _StaticDir:
        def __init__(self, sdir):
            self.name = sdir
        def cleanup(self):
            pass

    if not config.get("temp_dir_auto", True):
        tmp_dir = _StaticDir(temp_dir)
    else:
        tmp_dir = tempfile.TemporaryDirectory(dir=temp_dir, ignore_cleanup_errors=True)

    if peak_is_not_index:
        index_path = [os.path.join(tmp_dir.name, f"eval_{uuid.uuid4().hex}")]
    else:
        index_path = peak_filenames
        peak_filenames = None

    valid_charge = np.arange(1, config["max_charge"] + 1)
    dataloader_params = dict(
        batch_size=config["predict_batch_size"],
        n_peaks=config["n_peaks"],
        min_mz=config["min_mz"],
        max_mz=config["max_mz"],
        min_intensity=config["min_intensity"],
        remove_precursor_tol=config["remove_precursor_tol"],
        n_workers=config["n_workers"],
        test_filenames=peak_filenames,
        test_index_path=index_path,
        annotated=False,
        valid_charge=valid_charge,
        mode="test",
    )

    dataModule = DeNovoDataModule(**dataloader_params)
    dataModule.prepare_data()
    dataModule.setup(stage="test")
    test_dataloader = dataModule.test_dataloader()

    # Create the Trainer.
    if config.get("use_bf16") is True:
        logger.info("Using bf16 precision...")
        trainer = pl.Trainer(
            enable_model_summary=True,
            accelerator="auto",
            precision="bf16",
            auto_select_gpus=True,
            devices=_get_devices(),
            strategy=_get_strategy(),
        )
    else:
        logger.info("Using default precision...")
        trainer = pl.Trainer(
            enable_model_summary=True,
            accelerator="auto",
            auto_select_gpus=True,
            devices=_get_devices(),
            strategy=_get_strategy(),
        )

    # Run inference.
    run_trainer = trainer.predict
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model size: {pytorch_total_params} parameters")
    run_trainer(model, test_dataloader)

    tmp_dir.cleanup()


def _get_peak_filenames(path: str, supported_ext: Iterable[str] = (".mgf",)) -> List[str]:
    """Get all matching peak file names from the path pattern.

    Supports '&'-separated multiple path patterns for combining datasets.

    Parameters
    ----------
    path : str
        The path pattern (supports glob patterns and '&' separator).
    supported_ext : Iterable[str]
        Extensions of supported peak file formats.

    Returns
    -------
    List[str]
        The peak file names matching the path pattern.
    """
    if '&' in path:
        paths = path.split("&")
        files = []
        for p in paths:
            p = os.path.expanduser(p)
            p = os.path.expandvars(p)
            files += [fn for fn in glob.glob(p, recursive=True)]
        return files

    path = os.path.expanduser(path)
    path = os.path.expandvars(path)
    return [fn for fn in glob.glob(path, recursive=True)]


def _get_strategy() -> Optional[DDPStrategy]:
    """Get the DDP strategy for multi-GPU inference.

    Returns
    -------
    Optional[DDPStrategy]
        DDPStrategy if multiple GPUs are available, None otherwise.
    """
    if torch.cuda.device_count() > 1:
        return DDPStrategy(find_unused_parameters=False, static_graph=True)
    return None


def _get_devices() -> Union[int, str]:
    """Get the number of devices for inference.

    Returns
    -------
    Union[int, str]
        Number of GPUs to use, or "auto" for CPU-only.
    """
    import operator
    if any(
        operator.attrgetter(device + ".is_available")(torch)()
        for device in ["cuda", "backends.mps"]
    ):
        return -1
    elif not (n_workers := utils.n_workers()):
        return "auto"
    else:
        return n_workers
