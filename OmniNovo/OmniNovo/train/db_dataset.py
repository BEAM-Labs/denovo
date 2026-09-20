"""Dataset classes for loading mass spectra from LMDB databases."""
import torch
from typing import Optional, Tuple
from .db_index import DB_Index
import numpy as np
from torch.utils.data import Dataset
from loguru import logger


def cumsum(it):
    """Cumulative sum generator."""
    total = 0
    for x in it:
        total += x
        yield total


class DbDataset(Dataset):
    """Dataset that reads and processes spectra from multiple LMDB databases.

    Parameters
    ----------
    db_indexs : list of DB_Index
        The LMDB database indices to read from.
    n_peaks : int
        Maximum number of peaks to retain per spectrum.
    min_mz : float
        Minimum m/z value for peak filtering.
    max_mz : float
        Maximum m/z value for peak filtering.
    min_intensity : float
        Minimum relative intensity threshold (fraction of max intensity).
    remove_precursor_tol : float
        Tolerance in Da for removing precursor peaks.
    random_state : optional
        Random state for reproducibility.
    """

    def __init__(self, db_indexs, n_peaks: int = 150,
                 min_mz: float = 140.0,
                 max_mz: float = 2500.0,
                 min_intensity: float = 0.01,
                 remove_precursor_tol: float = 2.0,
                 random_state: Optional[int] = None):
        super().__init__()
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_intensity = min_intensity
        self.remove_precursor_tol = remove_precursor_tol
        self.rng = np.random.default_rng(random_state)
        self._indexs = db_indexs

    def __len__(self):
        return self.n_spectra

    def __getitem__(self, idx):
        for i, each_offset in enumerate(self.offset):
            if idx < each_offset:
                if i == 0:
                    new_idx = idx
                else:
                    new_idx = idx - self.offset[i - 1]

                mz_array, int_array, precursor_mz, precursor_charge, peptide, metadata = self.indexs[i][new_idx]
                spectrum = self._process_peaks(np.array(mz_array), np.array(int_array), precursor_mz, precursor_charge)
                break

        return spectrum, precursor_mz, precursor_charge, peptide, metadata

    def _process_peaks(
        self,
        mz_array: np.ndarray,
        int_array: np.ndarray,
        precursor_mz: float,
        precursor_charge: int,
    ) -> torch.Tensor:
        """Preprocess the spectrum: filter, normalize, and format peaks.

        Applies m/z range filtering, precursor peak removal, intensity filtering,
        top-K peak selection, root scaling, and L2 normalization.

        Parameters
        ----------
        mz_array : numpy.ndarray of shape (n_peaks,)
            The spectrum peak m/z values.
        int_array : numpy.ndarray of shape (n_peaks,)
            The spectrum peak intensity values.
        precursor_mz : float
            The precursor m/z.
        precursor_charge : int
            The precursor charge.

        Returns
        -------
        torch.Tensor of shape (n_peaks, 2)
            A tensor of the spectrum with the m/z and intensity peak values.
        """
        # 1. Type casting and sorting
        mz = np.asarray(mz_array, dtype=np.float64)
        inten = np.asarray(int_array, dtype=np.float32)

        if len(mz) > 0:
            order = np.argsort(mz)
            mz = mz[order]
            inten = inten[order]
        else:
            return torch.tensor([[0, 1]]).float()

        # 2. M/Z range filtering
        begin = np.searchsorted(mz, self.min_mz, side='left')
        end = np.searchsorted(mz, self.max_mz, side='right')
        mz = mz[begin:end]
        inten = inten[begin:end]

        if len(mz) == 0:
            return torch.tensor([[0, 1]]).float()

        # 3. Remove precursor peak
        if self.remove_precursor_tol > 0:
            adduct_mass = 1.007825
            neutral_mass = (precursor_mz - adduct_mass) * precursor_charge

            target_mzs = np.array([
                neutral_mass / charge + adduct_mass
                for charge in range(precursor_charge, 0, -1)
            ], dtype=np.float64)

            if len(target_mzs) > 0:
                diff = np.abs(mz[:, None] - target_mzs[None, :])
                mask = np.min(diff, axis=1) > self.remove_precursor_tol
                mz = mz[mask]
                inten = inten[mask]

        if len(mz) == 0:
            return torch.tensor([[0, 1]]).float()

        # 4. Filter by relative intensity threshold
        max_val = np.max(inten)
        threshold = self.min_intensity * max_val

        mask_noise = inten > threshold
        mz = mz[mask_noise]
        inten = inten[mask_noise]

        if len(mz) == 0:
            return torch.tensor([[0, 1]]).float()

        # 5. Top-K peak selection
        if self.n_peaks is not None and len(inten) > self.n_peaks:
            sort_idx = np.argsort(inten)[::-1]
            top_k_idx = sort_idx[:self.n_peaks]
            top_k_idx.sort()
            mz = mz[top_k_idx]
            inten = inten[top_k_idx]

        # 6. Root scaling and max normalization
        inten = np.sqrt(inten)
        inten = inten.astype(np.float32)

        max_i = np.max(inten)
        if max_i > 0:
            inten = inten * (1.0 / max_i).astype(np.float32)

        # 7. L2 normalization
        norm = np.linalg.norm(inten)
        if norm > 1e-9:
            inten = inten / norm

        return torch.tensor(np.stack([mz, inten], axis=1)).float()

    @property
    def offset(self):
        """Cumulative spectrum counts across all databases."""
        sizes_list = [each.n_spectra for each in self.indexs]
        return list(cumsum(sizes_list))

    @property
    def n_spectra(self) -> int:
        """The total number of spectra across all databases."""
        return sum(each.n_spectra for each in self.indexs)

    @property
    def indexs(self):
        """The underlying DB_Index objects."""
        return self._indexs

    @property
    def rng(self):
        """The NumPy random number generator."""
        return self._rng

    @rng.setter
    def rng(self, seed):
        """Set the NumPy random number generator."""
        self._rng = np.random.default_rng(seed)
