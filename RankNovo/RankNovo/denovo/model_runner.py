"""Training and testing functionality for the de novo peptide sequencing
model."""
import glob
import logging
import os
import tempfile
import uuid
from typing import Any, Dict, Iterable, List, Optional, Union
from tqdm import tqdm

import numpy as np
import torch

from .db_dataloader import DeNovoDataModule
from .msa_model import Spec2Pep

logger = logging.getLogger("RankNovo") 

import pickle


def evaluate(peak_path: str, model_filename: str, config: Dict[str,
                                                               Any]) -> None:
    """
    Evaluate peptide sequence predictions from a trained RankNovo model.

    Parameters
    ----------
    peak_path : str
        The path with peak files for predicting peptide sequences.
    model_filename : str
        The file name of the model weights (.ckpt file).
    config : Dict[str, Any]
        The configuration options.
    """
    print("Validating original predictions...")
    _execute_existing(peak_path, model_filename, config, True, save=config["save"])

def _execute_existing(
    peak_path: str,
    model_filename: str,
    config: Dict[str, Any],
    annotated: bool,
    out_writer=None,
    save=False
):
    """
    Predict peptide sequences with a trained RankNovo model with/without
    evaluation.

    Parameters
    ----------
    peak_path : str
        The path with peak files for predicting peptide sequences.
    model_filename : str
        The file name of the model weights (.ckpt file).
    config : Dict[str, Any]
        The configuration options.
    annotated : bool
        Whether the input peak files are annotated (execute in evaluation mode)
        or not (execute in prediction mode only).
    """
    
    # Read the MS/MS spectra for which to predict peptide sequences.
    if annotated:
        peak_ext = (".mgf", ".h5", ".hdf5")
    else:
        peak_ext = (".mgf", ".mzml", ".mzxml", ".h5", ".hdf5")
    print("peak file is: ", peak_path )
    if len(peak_filenames := _get_peak_filenames(peak_path, peak_ext)) == 0:
        logger.error("Could not find peak files from %s", peak_path)
        raise FileNotFoundError("Could not find peak files")
    peak_is_not_index = any(
        [os.path.splitext(fn)[1] in (".mgf", ".mzxml", ".mzml") for fn in peak_filenames])
    
    tmp_dir = tempfile.TemporaryDirectory()
    if peak_is_not_index:
        index_path = [os.path.join(tmp_dir.name, f"eval_{uuid.uuid4().hex}")]
    else:
        index_path = peak_filenames
        peak_filenames = None
    print("is peak not index?, ", peak_is_not_index)
    
    # index_path = [peak_path]
    # peak_filenames = None

    # SpectrumIdx = AnnotatedSpectrumIndex if annotated else SpectrumIndex
    valid_charge = np.arange(1, config["max_charge"] + 1)
    dataloader_params = dict(
        batch_size=config["predict_batch_size"],
        n_peaks=config["n_peaks"],
        min_mz=config["min_mz"],
        max_mz=config["max_mz"],
        min_intensity=config["min_intensity"],
        remove_precursor_tol=config["remove_precursor_tol"],
        n_workers=config["n_workers"],
        train_filenames = None,
        val_filenames = None,
        test_filenames = peak_filenames,
        train_index_path = None, # always a list, either a list containing one index path file or a list containing multiple db files 
        val_index_path = None,
        test_index_path = index_path,
        annotated = annotated,
        valid_charge = valid_charge , 
        mode = "test",
        by_train_path=config["by_train_path"],
        by_val_path=config["by_val_path"],
        by_test_path=config["by_test_path"],
        msa_test_path = config["msa_test_path"],
        model_id = config["model_id"]
    )
    # Initialize the data loader.
    dataModule = DeNovoDataModule(**dataloader_params)
    dataModule.prepare_data()
    dataModule.setup(stage="test")
    test_dataloader = dataModule.test_dataloader()
    
    # Load the trained model.
    if not os.path.isfile(model_filename):
        logger.error(
            "Could not find the trained model weights at file %s",
            model_filename,
        )
        raise FileNotFoundError("Could not find the trained model weights")
    model = Spec2Pep().load_from_checkpoint(
        model_filename,
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
        species=config["species"],
        use_col=config["use_col"]
    )

    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()


    total_count = 0
    total_aa_count = 0
    aa_precision_count = [0] * 10
    pep_recall_count = [0] * 10
    if save:
        new_msa = []
        new_TM = []
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dataloader)):
            
            aa_precision_list, n_aa, pep_recall_list, sequences, TM_predictions = model.my_pure_valid_forward(batch)
            total_count += len(batch)
            total_aa_count += n_aa
            
            for j, (aa, pep) in enumerate(zip(aa_precision_list, pep_recall_list)):
                aa_precision_count[j] += aa * n_aa
                pep_recall_count[j] += pep * len(batch)
            if save:
                new_msa.extend(sequences.tolist())
                new_TM.extend(TM_predictions.tolist())

            
    print("===============================================")
    for j, (aa, pep) in enumerate(zip(aa_precision_count, pep_recall_count)):
        if aa > 0 and pep > 0:
            print(f"model {j}: AA Precision: {aa / total_aa_count}, Pep Recall: {pep / total_count}")
    print("===============================================")
    if save:
        new_data = {
            "msa": new_msa,
            "TM": new_TM
        }
        species = peak_path.split('/')[-1].split('.')[0]
        with open(f"./new_msa_{species}.pkl", "wb") as f:
            pickle.dump(new_data, f)
    return None

def _get_peak_filenames(
    path: str, supported_ext: Iterable[str] = (".mgf", )) -> List[str]:
    """
    Get all matching peak file names from the path pattern.

    Performs cross-platform path expansion akin to the Unix shell (glob, expand
    user, expand vars).

    Parameters
    ----------
    path : str
        The path pattern.
    supported_ext : Iterable[str]
        Extensions of supported peak file formats. Default: MGF.

    Returns
    -------
    List[str]
        The peak file names matching the path pattern.
    """
    path = os.path.expanduser(path)
    path = os.path.expandvars(path)
    return [
        fn for fn in glob.glob(path, recursive=True)
        #if os.path.splitext(fn.lower())[1] in supported_ext
    ]
