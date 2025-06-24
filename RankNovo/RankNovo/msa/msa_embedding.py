import torch 
import torch.nn as nn 
import itertools

from ..components.encoders import MassEncoder
from .tokenizer import Alphabet

class MSAPeptideEmbedder(nn.Module):
    def __init__(
        self,
        dim_model,
        alphabet: Alphabet,
        max_charge=5,
    ):
        super().__init__()
        self.alphabat = alphabet
        self.aaDim = dim_model - dim_model // 2
        # precursor encoder
        self.mass_encoder = MassEncoder(dim_model // 2)
        self.charge_encoder = torch.nn.Embedding(max_charge, dim_model // 2)
        # aa encoder
        self.prefixMassEncoder = MassEncoder(dim_model // 4)
        self.suffixMassEncoder = MassEncoder(dim_model // 4)
        self.aa_encoder = torch.nn.Embedding(
            len(self.alphabat.all_toks),
            self.aaDim,
            padding_idx=self.alphabat.padding_idx,
        )

    @property
    def device(self):
        """The current device for the model"""
        return next(self.parameters()).device
    
    def forward(self, tokens, precursors):
        # tokens: (B, N, L), precursors: (B, 3)
        preMasses = self.deMass(tokens)
        suffixMasses = self.get_suffix_mass(tokens, precursors[:, 0])
        # precursors
        
        masses = self.mass_encoder(precursors[:, None, [0]])
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        precursors = masses + charges[:, None, :]

        # mass
        
        preAndSufPrecursors = torch.tensor([[0]]).to(self.device)
        preAndSufPrecursors = self.prefixMassEncoder(preAndSufPrecursors)
        preAndSufPrecursors = preAndSufPrecursors.repeat(precursors.shape[0],1)
        preAndSufPrecursors = preAndSufPrecursors.unsqueeze(1)
        precursors = torch.cat([precursors,preAndSufPrecursors,preAndSufPrecursors],dim=2)
    
        # tokens
        
        tgt = self.aa_encoder(tokens.to(torch.long))
        preMasses = preMasses.unsqueeze(-1)
        preMasses = self.prefixMassEncoder(preMasses)
        suffixMasses = suffixMasses.unsqueeze(-1)
        suffixMasses = self.suffixMassEncoder(suffixMasses)
        allMasses = torch.concat([preMasses,suffixMasses],dim=-1)
        tgtTemp = torch.concat([tgt,allMasses], dim=-1) # B, N, L, D
        B, N, L, D = tgtTemp.shape
        precursors = precursors.expand(B, N, D)
        
        tgtTemp[:, :, 0, :] += precursors
        return tgtTemp

    def deMass(self, tokens):
        tokens_list = tokens.cpu().tolist()
        mapped_tokens = [[[self.alphabat.idx_to_mass[element] for element in sublist] 
                                        for sublist in subarray] 
                                        for subarray in tokens_list]
        return torch.tensor(mapped_tokens).to(self.device)
    
    def get_suffix_mass(self, tokens, premass):
        premass = premass.cpu()
        tokens_list = tokens.cpu().tolist()
        suffix_mass = []
        for subarray in tokens_list:
            msa_suffix_mass = []
            for sublist in subarray:
                mass_list = [self.alphabat.idx_to_mass[idx] for idx in sublist]
                mass_list = list(itertools.accumulate(mass_list))
                msa_suffix_mass.append(mass_list)
               
            suffix_mass.append(msa_suffix_mass)
        suffix_mass = torch.tensor(suffix_mass)
        B, N, L = suffix_mass.shape
        premass = premass.unsqueeze(-1).unsqueeze(-1).expand(B, N, L)
        return (premass - suffix_mass).to(self.device)