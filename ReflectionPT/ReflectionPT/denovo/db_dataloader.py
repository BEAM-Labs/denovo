import functools
import os
import random
from typing import List, Optional, Tuple
from torch.utils.data import RandomSampler
import numpy as np
import pytorch_lightning as pl
import torch
from .db_index import DB_Index
from .db_dataset import DbDataset
class MyRandomSampler(torch.utils.data.Sampler):
    #randomly select N_1 and N_2 for ft
    def __init__(self, data_source, N_1, N_2, replacement=True):
        self.data_source = data_source
        self.replacement=replacement
        self.N_1 = N_1
        self.N_2 = N_2
    @property
    def num_samples(self):
        return self.N_1 + self.N_2
    def __iter__(self):
        offset = self.data_source.offset
        print("offset in sampler: ", offset)
        rand_tensor1 = torch.randint(low = 0, high = offset[0]-1, size = (self.N_1,) ).tolist()
        rand_tensor2 = torch.randint(low = offset[0], high = offset[1]-1, size = (self.N_2,)).tolist()
        rand_tensor1.extend(rand_tensor2)
        random.shuffle(rand_tensor1)
        return iter(rand_tensor1)
    
    def __len__(self):
        return self.num_samples
        
class MyRandomSampler_adapt(torch.utils.data.Sampler):
    #fix N1 sampled data and only sample N_2
    #N1 is our new dataset 
    def __init__(self, data_source, N_1, N_2, replacement=True):
        self.data_source = data_source
        self.replacement=replacement
        offset = self.data_source.offset
        self.N_1 = N_1
        self.N_2 = N_2
        #torch.manual_seed(0)
        self.N_1_range =  torch.randint(low = 0, high = offset[0]-1, size = (self.N_1,) ).tolist()
        
    @property
    def num_samples(self):
        return self.N_1 + self.N_2
    def __iter__(self):
        offset = self.data_source.offset
        #print("offset in sampler: ", offset)
        #rand_tensor1 = torch.randint(low = 0, high = offset[0]-1, size = (self.N_1,) ).tolist()
        rand_tensor2 = torch.randint(low = offset[0], high = offset[1]-1, size = (self.N_2,)).tolist()
        self.N_1_range.extend(rand_tensor2)
        random.shuffle(self.N_1_range)
        return iter(self.N_1_range)
    
    def __len__(self):
        return self.num_samples
        
  
class DeNovoDataModule(pl.LightningDataModule):
    #mange all the dataset (train, val, and test) and load the dataloder for training
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
        '''
        self.train_index: List[the DB_Index object]
        storing all the indexes that use to train the model, initialized to None
        
        self.valid_index: List[the DB_Index object]
        storing all the indexes that use to validate the model, initialized to None
        
        self.test_index: List[the DB_Index object]
        storing all the indexes that use to test the model, initialized to None
        
        
        self.train_filenames: List[Str]
        self.val_filenames: List[Str]
        self.test_filenames: List[Str]
        the list of mgf/mzxml/mxml filenames to load to DB index
        
        mode: if this Module is used in training or predicting 
        can either be "fit" or "test"
        
        
        
        
        
        
        
        '''
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
        '''
        set the self.train_dataset, self.val_dataset, self.test_dataset to the correct set of DB_Index
        this method will run on all GPUs 
        before run this method, make sure run PrepareData, which will enture the DB_index file is pre-created/updated
        and this method will link the processed DB_Index FIles to self.dataset
        '''
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
        '''
        This method will preprea the Index file upfront, in case we process it on multiple_GPUs during training'
        this method will get called only once by Lightning
        
        avoid using self.xx = xx since it won't get updated to all GPUs version
        
        
        
        
        '''
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
            #shuffle = True,
            pin_memory=True,
            num_workers=self.n_workers,
            sampler = sampler
        )
         
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the training DataLoader."""

        
        assert self.train_dataset != None
        M= 100000000 #498183   #Sample the training set if needed since some dataset is too large to finish entire epoch
        M =  1000000
        sampler = RandomSampler(self.train_dataset, replacement=True, num_samples=M)
        #sampler = None
        
        #sampler = MyRandomSampler_adapt(self.train_dataset, N_1 = 100000, N_2 = 1000000)
        #sampler = MyRandomSampler(self.train_dataset, N_1 = 10000, N_2 = 100000)
        return self._make_loader(self.train_dataset, sampler = sampler)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the validation DataLoader."""
        if self.mode == "fit":
            return [self._make_loader(self.valid_dataset), self._make_loader(self.test_dataset)]
        return self._make_loader(self.valid_dataset)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the test DataLoader."""
        #sampler = MyRandomSampler_adapt(self.test_dataset, N_1 = 50000, N_2 = 1)
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