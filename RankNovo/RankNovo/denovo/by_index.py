import lmdb
import numpy as np
import logging
from pathlib import Path
from .parser2 import  MzmlParser, MzxmlParser, MgfParser
import os 
import pickle

LOGGER = logging.getLogger(__name__)


class BY_Index:
    '''
    read and store and manage (read and write) a signle IMDB file 
    '''
    def __init__(self, db_path):
        #cant overwrite rn if there is already a db file in path
        index_path = Path(db_path)
        self.db_path = index_path
        
        self._init_db()
        
    def __getitem__(self, idx):
        with self.env.begin() as txn:
            # txn = self.env.begin()
            buffer = txn.get(str(idx).encode())
            data= pickle.loads(buffer)
            return data
    
    def __len__ (self):
        return self.env.stat()['entries']
                  
            
    def _init_db(self):

        self.env = lmdb.open(str(self.db_path), map_size=int(1e9), subdir=True, readonly=True, lock=False)
        