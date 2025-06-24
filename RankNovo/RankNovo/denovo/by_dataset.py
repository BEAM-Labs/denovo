import torch
from typing import Optional, Tuple, List
from .db_index import DB_Index
import numpy as np
import spectrum_utils.spectrum as sus
from torch.utils.data import Dataset

class byDataset(Dataset):
    '''
    Read and Write and manage MSA data
    '''
    def __init__(self, DbDataset, lable, msa, mode="train", model_id=None):
        super().__init__()
        self.DbDataset = DbDataset
        self.lable = lable
        self.msa = msa
     
        self.mode = mode
        self.model_id = model_id
        match model_id:
            case "model_6":
                self.msa_mask = [0,1,2,3,4,5,6]
            case "model_5":
                self.msa_mask = [0,1,2,4,5,6]
            case "model_4":
                self.msa_mask = [0,1,4,5,6]
            case "model_3":
                self.msa_mask = [0,1,4,6]
            case "model_2":
                self.msa_mask = [0,1,4]
            case _:
                self.msa_mask = None

       

    def __len__(self):
        return len(self.DbDataset)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, float, int, str, torch.tensor]:
        
        data = list(self.DbDataset[idx])

        # msa
        msa = self.msa[idx]
       
        for p in msa:
            p = p.replace("pyro-","-17.027")
            p = p.replace("I", "L")


        msa = np.array(msa)
        if self.msa_mask is not None:
            msa = msa[self.msa_mask].tolist()
        else:
            msa = msa.tolist()

        data[3] = msa

        # by is None
        if self.lable is None:
            x = data[:4]
            x.append(self.lable)
            return x
        
        if torch.equal(data[0], torch.tensor([[0, 1]]).float()):
            bylable = [0]
        else:
            bylable = np.array(self.lable[idx][data[4]:data[5]])
            bylable = bylable[data[6]]
            bylable = bylable[data[7]]
        
        x = data[:4]
        x.append(bylable)

        # print(data[0].shape)
        # print(len(bylable))
        assert data[0].shape[0] == len(bylable), f"{idx} wrong"

        return x

    

    