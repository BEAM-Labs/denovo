import torch
from typing import Optional, Tuple
from .db_index import DB_Index
import numpy as np
import spectrum_utils.spectrum as sus
from torch.utils.data import Dataset
def cumsum(it):
    total = 0
    for x in it:
        total += x
        yield total
class DbDataset(Dataset):
    '''
    Read and Write and manage multiple DB files and process the data from those files 
    '''
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
    
    def __getitem__(self,idx):
        for i, each_offset in enumerate(self.offset):
            if idx < each_offset:
                if i == 0:
                    new_idx = idx
                else:
                    new_idx = idx - self.offset[i-1]
                    
                
                mz_array, int_array, precursor_mz, precursor_charge, peptide = self.indexs[i][new_idx]
                
                
                spectrum = self._process_peaks(np.array(mz_array), np.array(int_array), precursor_mz, precursor_charge)
                break
        return spectrum, precursor_mz, precursor_charge, peptide

    def _process_peaks(
        self,
        mz_array: np.ndarray,
        int_array: np.ndarray,
        precursor_mz: float,
        precursor_charge: int,
    ) -> torch.Tensor:
        """
        Preprocess the spectrum by removing noise peaks and scaling the peak
        intensities.

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
        spectrum = sus.MsmsSpectrum(
            "",
            precursor_mz,
            precursor_charge,
            mz_array.astype(np.float64),
            int_array.astype(np.float32),
        )
        try:
            spectrum.set_mz_range(self.min_mz, self.max_mz)
            if len(spectrum.mz) == 0:
                raise ValueError
            spectrum.remove_precursor_peak(self.remove_precursor_tol, "Da")
            if len(spectrum.mz) == 0:
                raise ValueError
            spectrum.filter_intensity(self.min_intensity, self.n_peaks)
            if len(spectrum.mz) == 0:
                raise ValueError
            spectrum.scale_intensity("root", 1)
            intensities = spectrum.intensity / np.linalg.norm(
                spectrum.intensity
            )
            return torch.tensor(np.array([spectrum.mz, intensities])).T.float()
        except ValueError:
            # Replace invalid spectra by a dummy spectrum.
            return torch.tensor([[0, 1]]).float()            
        
        
    @property
    def offset(self):
        sizes_list = []
        for each in self.indexs:
            sizes_list.append(each.n_spectra)
        return list(cumsum(sizes_list))
            
        
            
        
    @property
    def n_spectra(self) -> int:
        """The total number of spectra."""
        total = 0
        for each in self.indexs:
            print("each data size: ", each.n_spectra)
            total += each.n_spectra
        return total
    @property
    def indexs(self):
        """The underlying SpectrumIndex."""
        return self._indexs
    @property
    def rng(self):
        """The NumPy random number generator."""
        return self._rng
    @rng.setter
    def rng(self, seed):
        """Set the NumPy random number generator."""
        self._rng = np.random.default_rng(seed)