"""Data loading and collation utilities for de novo peptide inference."""
import functools
import os
from typing import List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from .db_index import DB_Index
from .db_dataset import DbDataset
from loguru import logger


class DeNovoDataModule(pl.LightningDataModule):
    """Lightning DataModule for reading spectra used by inference."""

    def __init__(
        self,
        test_index=None,
        batch_size: int = 128,
        n_peaks: Optional[int] = 150,
        min_mz: float = 50.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,
        n_workers: Optional[int] = None,
        random_state: Optional[int] = None,
        test_filenames=None,
        test_index_path=None,
        annotated=False,
        valid_charge=None,
        ms_level=2,
        **_ignored,
    ):
        super().__init__()
        self.annotated = annotated
        self.valid_charge = valid_charge
        self.ms_level = ms_level
        self.test_index = test_index
        self.batch_size = batch_size
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_intensity = min_intensity
        self.remove_precursor_tol = remove_precursor_tol
        self.n_workers = n_workers if n_workers is not None else os.cpu_count()
        self.rng = np.random.default_rng(random_state)
        self.test_dataset = None
        self.test_filenames = test_filenames
        self.test_index_path = test_index_path

    def setup(self, stage=None):
        """Link the prepared spectrum index files to a test dataset."""
        if stage not in (None, "test", "predict"):
            return
        make_dataset = functools.partial(
            DbDataset,
            n_peaks=self.n_peaks,
            min_mz=self.min_mz,
            max_mz=self.max_mz,
            min_intensity=self.min_intensity,
            remove_precursor_tol=self.remove_precursor_tol,
        )
        if self.test_index is None:
            self.test_index = []
            for each in self.test_index_path or []:
                self.test_index.append(DB_Index(
                    each, None, self.ms_level, self.valid_charge, self.annotated, lock=False
                ))
        self.test_dataset = make_dataset(self.test_index, random_state=self.rng)

    def prepare_data(self) -> None:
        """Create missing indexed spectrum files from the supplied input."""
        if self.test_index is not None:
            return
        self.test_index = []
        for each in self.test_index_path or []:
            self.test_index.append(DB_Index(
                each, self.test_filenames, self.ms_level, self.valid_charge, self.annotated,
                lock=self.test_filenames is not None,
            ))

    def _make_loader(self, dataset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=prepare_batch,
            shuffle=False,
            pin_memory=False,
            num_workers=max(0, self.n_workers // 2),
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("The spectrum dataset has not been initialized")
        return self._make_loader(self.test_dataset)

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return self.test_dataloader()


def prepare_batch(
    batch: List[Tuple[torch.Tensor, float, int, str, dict]]
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, List[dict]]:
    """Collate MS/MS spectra into a padded batch."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        raise ValueError("All elements in the batch are None/Corrupted!")

    spectra, precursor_mzs, precursor_charges, spectrum_ids, metadata_list = list(zip(*batch))
    spectra = torch.nn.utils.rnn.pad_sequence(spectra, batch_first=True)
    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - 1.007276) * precursor_charges
    precursors = torch.vstack([precursor_masses, precursor_charges, precursor_mzs]).T.float()
    return spectra, precursors, np.asarray(spectrum_ids), list(metadata_list)
