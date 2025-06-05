import functools
import os
from typing import List, Optional, Tuple
from torch.utils.data import RandomSampler
import numpy as np
import pytorch_lightning as pl
import torch
from .db_index import DB_Index
from .db_dataset import DbDataset
class DeNovoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_index= None, # list 
        valid_index=  None, #list 
        test_index=  None, # list 
        batch_size: int = 128,
        n_peaks: Optional[int] = 150,
        min_mz: float = 50.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,
        n_workers: Optional[int] = None,
        random_state: Optional[int] = None,
        train_filenames = None,
        val_filenames = None,
        test_filenames = None,
        train_index_path = None,
        val_index_path = None,
        test_index_path = None,
        annotated = True,
        valid_charge = None ,
        ms_level = 2, 
        mode = "fit"
        
    ): 
        super().__init__()
        self.annotated = annotated
        self.valid_charge = valid_charge
        self.ms_level = ms_level
        self.train_index = train_index 
        self.valid_index = valid_index
        self.test_index = test_index
        self.batch_size = batch_size
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_intensity = min_intensity
        self.remove_precursor_tol = remove_precursor_tol
        self.n_workers = n_workers if n_workers is not None else os.cpu_count()
        self.rng = np.random.default_rng(random_state)
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.train_filenames = train_filenames #either a list or None, need to examine 
        self.val_filenames = val_filenames
        self.test_filenames = test_filenames
        self.train_index_path = train_index_path # always a list, one or more values in the list 
        self.val_index_path = val_index_path 
        self.test_index_path = test_index_path
        self.mode = mode
    def setup(self, stage=None):
        if stage in (None, "fit", "validate"):
            make_dataset = functools.partial(
                DbDataset,
                n_peaks=self.n_peaks,
                min_mz=self.min_mz,
                max_mz=self.max_mz,
                min_intensity=self.min_intensity,
                remove_precursor_tol=self.remove_precursor_tol,
            )
            self.train_index = []
            for each in self.train_index_path:
                self.train_index.append(DB_Index(each, None, self.ms_level, self.valid_charge, self.annotated, lock=False))
            self.train_dataset = make_dataset(self.train_index, random_state=self.rng )
            
            print("index_len:", len(self.train_index[0]))
            
            self.valid_index = []
            for each in self.val_index_path:
                self.valid_index.append(DB_Index(each, None, self.ms_level, self.valid_charge, self.annotated, lock=False))
            
            self.valid_dataset = make_dataset(self.valid_index, random_state= self.rng)
            
            self.test_index = []
            for each in self.test_index_path:
                self.test_index.append(DB_Index(each, None, self.ms_level, self.valid_charge, self.annotated, lock=False))
            self.test_dataset = make_dataset(self.test_index)
            
            
            
        elif stage in ( "test"):
            make_dataset = functools.partial(
                DbDataset,
                n_peaks=self.n_peaks,
                min_mz=self.min_mz,
                max_mz=self.max_mz,
                min_intensity=self.min_intensity,
                remove_precursor_tol=self.remove_precursor_tol,
            )
            self.test_index = []
            for each in self.test_index_path:
                self.test_index.append(DB_Index(each, None, self.ms_level, self.valid_charge, self.annotated, lock = False))
            self.test_dataset = make_dataset(self.test_index)
            
            
    def prepare_data(self) -> None:
        #rule: if db_index file is None, we create index using filenames
        #      else: we ignore filenames!!!  
        # overall, if any filename is provided, we have to call prepare_data
        #need a mode for training/val/test
        print("prepare_data ing.....")
        
        
        if self.train_index == None and self.mode == "fit": #prepare train_index
            
            '''
            try:
                assert self.train_filenames != None
            except:
                raise ValueError("No training file provided ")
            '''
            if self.train_filenames == None :
                lock = False
            else:
                lock = True
            for each in self.train_index_path:
                DB_Index(each, self.train_filenames, self.ms_level, self.valid_charge, self.annotated, lock= lock)
        if self.valid_index == None and self.mode=="fit": # prepare val_index
            
            '''
            try:
                assert self.val_filenames != None
            except:
                raise ValueError("No validation file provided ")
            '''
            if self.val_filenames == None:
                lock = False
            else:
                lock = True
            for each in self.val_index_path:
                DB_Index(each, self.val_filenames, self.ms_level, self.valid_charge, self.annotated, lock=lock)
        if self.test_index == None :
            '''
            try:
                assert self.test_filenames != None
            except:
                raise ValueError("No training file provided ")
                
            '''
            if self.test_filenames == None:
                lock = False
            else:
                lock = True
            for each in self.test_index_path:
                DB_Index(each, self.test_filenames, self.ms_level, self.valid_charge, self.annotated, lock=lock)
        if  self.train_index != None:    # to be changed, add a checker for existance 
            pass

    def _make_loader(
        self, dataset: torch.utils.data.Dataset, sampler = None
    ) -> torch.utils.data.DataLoader:
         return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=prepare_batch,
            pin_memory=True,
            num_workers=self.n_workers,
            sampler = sampler
        )
         
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the training DataLoader."""

        
        assert self.train_dataset != None
        M= 898183 #498183
        sampler = RandomSampler(self.train_dataset, replacement=True, num_samples=M)
        return self._make_loader(self.train_dataset, sampler = sampler)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the validation DataLoader."""
        if self.mode == "fit":
            return [self._make_loader(self.valid_dataset), self._make_loader(self.test_dataset)]
        return self._make_loader(self.valid_dataset)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the test DataLoader."""
        return self._make_loader(self.test_dataset)

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the predict DataLoader."""
        return self._make_loader(self.test_dataset) 
         
def prepare_batch(
    batch: List[Tuple[torch.Tensor, float, int, str]]
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Collate MS/MS spectra into a batch.

    The MS/MS spectra will be padded so that they fit nicely as a tensor.
    However, the padded elements are ignored during the subsequent steps.

    Parameters
    ----------
    batch : List[Tuple[torch.Tensor, float, int, str]]
        A batch of data from an AnnotatedSpectrumDataset, consisting of for each
        spectrum (i) a tensor with the m/z and intensity peak values, (ii), the
        precursor m/z, (iii) the precursor charge, (iv) the spectrum identifier.

    Returns
    -------
    spectra : torch.Tensor of shape (batch_size, n_peaks, 2)
        The padded mass spectra tensor with the m/z and intensity peak values
        for each spectrum.
    precursors : torch.Tensor of shape (batch_size, 3)
        A tensor with the precursor neutral mass, precursor charge, and
        precursor m/z.
    spectrum_ids : np.ndarray
        The spectrum identifiers (during de novo sequencing) or peptide
        sequences (during training).
    """
    spectra, precursor_mzs, precursor_charges, spectrum_ids = list(zip(*batch))
    spectra = torch.nn.utils.rnn.pad_sequence(spectra, batch_first=True)
    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - 1.007276) * precursor_charges
    precursors = torch.vstack(
        [precursor_masses, precursor_charges, precursor_mzs]
    ).T.float()
    return spectra, precursors, np.asarray(spectrum_ids)