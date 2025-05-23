import lmdb
import numpy as np
import logging
from pathlib import Path
from .parser2 import  MzmlParser, MzxmlParser, MgfParser
import os 
import pickle

LOGGER = logging.getLogger(__name__)
def listify(obj):
    """Turn an object into a list, but don't split strings."""
    try:
        assert not isinstance(obj, str)
        iter(obj)
    except (AssertionError, TypeError):
        obj = [obj]

    return list(obj)
class DB_Index:
    '''
    read and store and manage (read and write) a signle IMDB file 
    '''
    def __init__(self, db_path, filenames=None, ms_level = 2, valid_charge = None, annotated = True, lock = True ):
        '''
        self.n_spectra is the current index to write 
        e.g, if self.n_spectra = 0, we update(0, spectra)
        '''
        #cant overwrite rn if there is already a db file in path
        index_path = Path(db_path)
        self.db_path = index_path
        self.lock = lock
        print("what lock is this:", lock)
        
        
        
        self.filenames = filenames
        self.ms_level=ms_level
        self.valid_charge=valid_charge
        self.annotated = bool(annotated)
        is_exist = index_path.exists()
        
        self._init_db()
        
        #create a new 
        if not is_exist:
            # print("I am not exist")
            txn = self.env.begin(write=True)
            txn.put( "ms_level".encode(), str(self.ms_level).encode())
            txn.put("n_spectra".encode(),  str(0).encode())
            txn.put("n_peaks".encode(), str(0).encode())
            self.n_spectra = 0
            # print("Before Writing:", self.annotated)
            # txn.put("annotated".encode(), str(int(self.annotated)).encode())
            # annotated_read = bool(txn.get("annotated".encode()).decode())
            # print("jinzhi:",annotated_read,self.annotated)
            txn.commit()
        else:
            txn=self.env.begin()
            self.n_spectra = int(txn.get("n_spectra".encode()).decode())
            ms_level_read = int(txn.get("ms_level".encode()).decode())
            # annotated_read = bool(txn.get("annotated".encode()).decode())
            try:
                assert ms_level_read == self.ms_level
                # assert annotated_read == self.annotated
            except:
                raise ValueError(f"{self.db_path} already existed, but it has a inconsistent ms_level/annotated with input parameter!")
        if filenames is not None:
            filenames = listify(filenames)
            LOGGER.info("Reading %i files...", len(filenames))
            for ms_file in filenames:
                self.add_file(ms_file)
                
                
    def write_to_db(self, parser, n ):
        #write all spectrums in the parser to lmdb
        txn = self.env.begin(write=True)
        assert n == len(parser.precursor_charge)
        for i in range(len(parser.precursor_charge)):
            #print("write in %i th spectrum,", i)
            #remember to round the charge 
            #precusor m/z, precusor charge, peaks number, m/z1, .... , m/zn, i_1, ...., i_n 
            collection_data = {"precursor_mz": parser.precursor_mz[i], "precursor_charge": parser.precursor_charge[i],
                               "mz_array": parser.mz_arrays[i], "intensity_array": parser.intensity_arrays[i]}
            
            collection_data["pep"] =  parser.annotations[i]
            # print(parser.annotations[i])
                
            buffers = pickle.dumps(collection_data)
            
            txn.put(str(self.n_spectra).encode(), buffers)
            self.n_spectra+=1
        txn.put("n_spectra".encode(),  str(self.n_spectra).encode())    
        txn.commit()
        
    def __getitem__(self, idx):
        txn = self.env.begin()
        buffer = txn.get(str(idx).encode())
        data= pickle.loads(buffer)
        pep = data["pep"]
        out = (data["mz_array"], data["intensity_array"],  data["precursor_mz"], data["precursor_charge"], pep)
        return out
    def __len__ (self):
        return self.n_spectra
           
    def add_file(self, a_file):
        a_file = Path(a_file)
        parser = self._get_parser(a_file)
        parser.read()
        n_spectra_in_file = parser.n_spectra
        self.write_to_db(parser, n_spectra_in_file)          
            
    def _init_db(self):
        if self.lock == True:
            read = False
        else:
            read = True
        self.env = lmdb.open(str(self.db_path), map_size=4099511627776, subdir=False, readonly = read, lock=self.lock)
        
        
    def _get_parser(self, ms_data_file):
        #to allow change of annotations
        kw_args = dict(ms_level=self.ms_level, valid_charge=self.valid_charge, annotationsLabel=self.annotated)
        if ms_data_file.suffix.lower() == ".mzml":
            return MzmlParser(ms_data_file, **kw_args)

        if ms_data_file.suffix.lower() == ".mzxml":
            return MzxmlParser(ms_data_file, **kw_args)

        if ms_data_file.suffix.lower() == ".mgf":
            return MgfParser(ms_data_file, **kw_args)

        raise ValueError("Only mzML, mzXML, and MGF files are supported.")