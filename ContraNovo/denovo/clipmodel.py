from collections import OrderedDict
# from typing import Tuple,Union
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn.functional as F
from torch import nn
from ..masses import PeptideMass
from .. import utils2 as utils
import itertools
import re

from ..components.encoders import MassEncoder,PeakEncoder,PositionalEncoder

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, key_padding_mask : torch.Tensor = None):
        key_padding_mask  = key_padding_mask .to(dtype=x.dtype, device=x.device) if key_padding_mask  is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask =key_padding_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        x = x + self.attention(self.ln_1(x),attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, dropout) for _ in range(layers)])

    def forward(self, x: torch.Tensor, key_padding_mask : torch.Tensor = None):
        for module in self.resblocks:
            x = module(x,key_padding_mask)
        return x


class SpectrumEncoder(nn.Module):
    def __init__(self,
                 embed_dim:int = 512,
                 #Spectrum:
                 n_peaks: int = 150,
                 transformer_width: int = 512,
                 transformer_heads: int = 8,
                 transformer_layers: int = 9,
                 max_charge = 10,
                 dropout = 0.18):
        super().__init__()
        
        self.transformer_width = transformer_width
        self.embed_dim = embed_dim 

        self.n_peaks = n_peaks

        self.mass_encoder = MassEncoder(transformer_width)

        # self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, embed_dim))

        # self.percursors_param = torch.nn.Parameter(torch.randn(transformer_width,transformer_width))

        layer = torch.nn.TransformerEncoderLayer(
            d_model=transformer_width,
            nhead=transformer_heads,
            dim_feedforward=1024,
            batch_first=True,
            dropout=dropout,
        )

        self.spectraTransformer = torch.nn.TransformerEncoder(
            layer,
            num_layers = transformer_layers,
        )


        # self.spectraTransformer = Transformer(
        #     width=transformer_width,
        #     layers=transformer_layers,
        #     heads=transformer_heads,
        #     dropout=dropout
        # )

        self.peak_encoder = PeakEncoder(
            transformer_width,
            dim_intensity=None    
        )

        #Precursor Encoder
        self.mass_encoder = MassEncoder(self.transformer_width)
        self.charge_encoder = torch.nn.Embedding(max_charge, transformer_width)

        

        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self,spectra,precursors):
        '''---------------------------------------------------------------'''
        # Transformer Encoder for Peaks and Precursor.

        # masses = self.mass_encoder(precursors[:, None, [0]])
        # charges = self.charge_encoder(precursors[:, 1].int() - 1)
        # precursors = masses + charges[:, None, :]

        zeros = ~spectra.sum(dim=2).bool()

        # mask = [
        #     # torch.tensor([[False]] * spectra.shape[0]).type_as(zeros),
        #     # torch.tensor([[False]] * spectra.shape[0]).type_as(zeros),
        #     zeros,
        # ]
        # mask = torch.cat(mask, dim=1)
        mask = zeros

        peaks = self.peak_encoder(spectra)

        # precursors = torch.matmul(precursors,self.percursors_param)
        # peaks = torch.concat([precursors,peaks],dim = 1)

        # Add the spectrum representation to each input:
        # latent_spectra = self.latent_spectrum.expand(peaks.shape[0], -1, -1)

        # peaks = torch.cat([latent_spectra, peaks], dim=1)

        # peaks = peaks + precursors

        # Peaks after Transformer Encoder named as pkt 
        # pkt = peaks.permute(1,0,2) # Do premute because the batch_is_first of SelfAttentionLayer is False
        pkt = self.spectraTransformer(peaks,src_key_padding_mask=mask)
        # pkt = pkt.permute(1,0,2) #Shape(B_s, Specturm_len, Feature_size)

        return pkt, mask



class PeptideEncoder(nn.Module):
    def __init__(self,
                 #Peptide:
                 max_len = 100,
                 residues: Union[Dict[str, float], str] = "canonical",
                 ptransformer_width: int = 512,
                 ptransformer_heads: int = 8,
                 ptransformer_layers: int = 9,
                 dropout = 0.4):
        
        super().__init__()

        self.reverse = True

        self.max_len = max_len
        self._peptide_mass = PeptideMass(residues=residues)
        self._amino_acids = list(self._peptide_mass.masses.keys())
        self._idx2aa = {i + 1: aa for i, aa in enumerate(self._amino_acids)}
        self._aa2idx = {aa: i for i, aa in self._idx2aa.items()}

        self.aminoEmbedDim = ptransformer_width - 256
        self.aa_encoder = torch.nn.Embedding(
            len(self._amino_acids) + 1,
            self.aminoEmbedDim,
            padding_idx=0,
        )

        #Mass/charge Encoder for precursors (Dim:256 = dim_model // 2)
        self.mass_encoder = MassEncoder(256)
        self.charge_encoder = torch.nn.Embedding(10, 256)

        # MassEncoder for prefix and suffix mass
        self.prefixMassEncoder = MassEncoder(128)
        self.suffixMassEncoder = MassEncoder(128)

        self.massEncoder = MassEncoder(ptransformer_width)

        self.pos_encoder = PositionalEncoder(ptransformer_width)

        layer = torch.nn.TransformerEncoderLayer(
            d_model=ptransformer_width,
            nhead=ptransformer_heads,
            dim_feedforward=1024,
            batch_first=True,
            dropout=dropout,
        )

        self.peptideTransformer = torch.nn.TransformerEncoder(
            layer,
            num_layers = ptransformer_layers,
        )

        # self.peptideTransformer = Transformer(
        #     width=ptransformer_width,
        #     layers=ptransformer_layers,
        #     heads=ptransformer_heads,
        #     dropout=dropout
        # )

        #Massses Embedding
        '''self.linearEmbedding = torch.nn.Linear(1,2)'''
      
    
    def forward(self,sequences,precursors):

        # Transformer Encoder For Peptide Sequence.
        # # Mass Encoder For Peptide Sequence.
        '''if sequences is not None:
            sequences = utils.listify(sequences)
            tokens = [self.tokenize(s) for s in sequences]
            Masses = [self.deMass(s) for s in sequences]
            Masses = torch.nn.utils.rnn.pad_sequence(Masses, batch_first = True)
            tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first = True)
        else:
            tokens = torch.tensor([[]]).to(self.device)
            Masses = torch.tensor([[]]).to(self.device) 

        tgt = self.aa_encoder(tokens)

        Masses = Masses.unsqueeze(2)
        Masses = self.linearEmbedding(Masses)
        tgt = torch.concat([tgt,Masses],dim=2)

        tgt = self.pos_encoder(tgt)'''
        if sequences is not None:
            sequences = utils.listify(sequences)
            Masses = [self.deMass(sequences[i]) for i in range(len(sequences))]
            Masses = torch.nn.utils.rnn.pad_sequence(Masses, batch_first = True)
            suffixMasses = [self.get_suffix_mass(sequences[i],precursors[i][0]) for i in range(len(sequences))]
            suffixMasses = torch.nn.utils.rnn.pad_sequence(suffixMasses, batch_first = True)
            tokens = [self.tokenize(s) for s in sequences]
            tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first = True)
        else:
            Masses = torch.tensor([[]]).to(self.device)
            suffixMasses = torch.tensor([[]]).to(self.device)
            tokens = torch.tensor([[]]).to(self.device)


        masses = self.mass_encoder(precursors[:, None, [0]])
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        precursors = masses + charges[:, None, :]

        preAndSufPrecursors = torch.tensor([[0]]).to(self.device)
        preAndSufPrecursors = self.prefixMassEncoder(preAndSufPrecursors)
        preAndSufPrecursors = preAndSufPrecursors.repeat(precursors.shape[0],1)
        preAndSufPrecursors = preAndSufPrecursors.unsqueeze(1)
        precursors = torch.cat([precursors,preAndSufPrecursors,preAndSufPrecursors],dim=2)

        

        Masses = Masses.unsqueeze(2)
        Masses = self.prefixMassEncoder(Masses)

        suffixMasses = suffixMasses.unsqueeze(2)
        suffixMasses = self.suffixMassEncoder(suffixMasses)

        Masses = torch.concat([Masses,suffixMasses],dim=2)

        tgt = self.aa_encoder(tokens)
        tgt_key_padding_mask = tgt.sum(axis=2) == 0
        tgt = torch.concat([tgt,Masses],dim=2)

        tgt = torch.cat([precursors, tgt], dim=1)

        tgt_key_padding_mask = tgt.sum(axis=2) == 0
        # Add positional code on peptide sequence.
        # tgt = self.pos_encoder(tgt) #(n_spectra, len(Peptide), dim_model)
        # Peptide input to Transformer.

        # tgt = tgt.permute(1,0,2)
        tgt = self.peptideTransformer(tgt,src_key_padding_mask = tgt_key_padding_mask)
        # tgt = tgt.permute(1,0,2) #Shape(B_s, Peptide_len, Feature_size)        

        return tgt
          
    
    def tokenize(self, sequence, partial=False):
        """Transform a peptide sequence into tokens

        Parameters
        ----------
        sequence : str
            A peptide sequence.

        Returns
        -------
        torch.Tensor
            The token for each amino acid in the peptide sequence.
        """
        if not isinstance(sequence, str):
            return sequence  # Assume it is already tokenized.

        sequence = sequence.replace("I", "L")
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)

        if self.reverse:
            sequence = list(reversed(sequence))

        # if not partial:
        #     sequence += ["$"]

        tokens = [self._aa2idx[aa] for aa in sequence]
        tokens = torch.tensor(tokens, device=self.device)
        return tokens

    def deMass(self,sequence):

        if not isinstance(sequence, str):

            sequence = [self._idx2aa.get(i.item(), "") for i in sequence]
            masses = [self._peptide_mass.masses[aa] for aa in sequence]
            masses = list(itertools.accumulate(masses))
            masses = torch.tensor(masses, device = self.device)

            return masses
        
        sequence = sequence.replace("I", "L")
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)

        if self.reverse:
            sequence = list(reversed(sequence))

        masses = [self._peptide_mass.masses[aa] for aa in sequence]

        masses = list(itertools.accumulate(masses))

        masses = torch.tensor(masses, device = self.device)

        return masses
    
    def get_suffix_mass(self,sequence,premass):

        if not isinstance(sequence, str):

            sequence = [self._idx2aa.get(i.item(), "") for i in sequence]
            masses = [self._peptide_mass.masses[aa] for aa in sequence]
            masses = list(itertools.accumulate(masses))
            masses = torch.tensor(masses, device = self.device)
            masses = premass - masses
            return masses
        
        sequence = sequence.replace("I", "L")
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)

        if self.reverse:
            sequence = list(reversed(sequence))

        masses = [self._peptide_mass.masses[aa] for aa in sequence]

        masses = list(itertools.accumulate(masses))

        masses = torch.tensor(masses, device = self.device)
        masses = premass - masses

        return masses
    
    def get_mass(self, sequence):

        if not isinstance(sequence, str):
            masses = [self._peptide_mass.masses[aa] for aa in sequence]
            masstemp = torch.tensor(masses, device = self.device)
            return masstemp
        
        sequence = sequence.replace("I", "L")
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)

        if self.reverse:
            sequence = list(reversed(sequence))

        masses = [self._peptide_mass.masses[aa] for aa in sequence]
        masstemp = torch.tensor(masses, device = self.device)
        # masstemp = torch.cat([masstemp,torch.tensor([0.0]).to(masstemp.device)])

        return masstemp
    
    def getAminoAcid(self):
        AA_masslist = [self._peptide_mass.masses[self._idx2aa[i]] for i in range(1,28)]
        AA_masslist = [0] + AA_masslist
        AA_masslist = torch.tensor(AA_masslist,device = self.device)
        return AA_masslist
        
    @property
    def device(self):
        """The current device for the model"""
        return next(self.parameters()).device
    
class CLIP(nn.Module):
    def __init__(self,
                 embed_dim:int = 512,
                 #Spectrum:
                 n_peaks: int = 150,
                 transformer_width: int = 512,
                 transformer_heads: int = 8,
                 transformer_layers: int = 9,
                 #Peptide:
                 max_len = 100,
                 residues: Union[Dict[str, float], str] = "canonical",
                 ptransformer_width: int = 512,
                 ptransformer_heads: int = 8,
                 ptransformer_layers: int = 9,
                 max_charge = 10,
                 ):
        super().__init__()
        
        self.spectrumEncoder = SpectrumEncoder(embed_dim = embed_dim,
                 #Spectrum:
                 n_peaks = n_peaks,
                 transformer_width = transformer_width,
                 transformer_heads = transformer_heads,
                 transformer_layers = transformer_layers,
                 max_charge = max_charge,
                 )
        self.peptideEncoder = PeptideEncoder(max_len = max_len,
                 residues = residues,
                 ptransformer_width = ptransformer_width,
                 ptransformer_heads = ptransformer_heads,
                 ptransformer_layers = ptransformer_layers)
        
        self.global_peptide = torch.nn.Parameter(torch.randn(1,1,ptransformer_width))
        # Which be used to calc the global feature vector for spectrum.
        self.global_spectrum = torch.nn.Parameter(torch.randn(1,1,transformer_width))

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_spectrum(self,spectra,precursors):

        pkt = self.spectrumEncoder(spectra,precursors)

        '''-----------------------------------------------------'''
        # Extract the global features for Spectrum and Peptide. 

        '''Sprectrum Global Features:'''
        pkt = torch.transpose(pkt,1,2)
        ratiospkt = torch.matmul(self.global_spectrum, pkt)
        ratiospkt = torch.softmax(ratiospkt,dim = 2)
        pkt = torch.transpose(pkt,1,2)
        pkt = torch.matmul(ratiospkt,pkt)
        pkt = pkt.squeeze(1)

        return pkt
    
    def encode_peptide(self,sequences):

        tgt =  self.peptideEncoder(sequences)
        
        '''Peptide Global Features:'''

        tgt = torch.transpose(tgt,1,2)
        ratiostgt = torch.matmul(self.global_peptide, tgt)
        ratiostgt = torch.softmax(ratiostgt,dim = 2)
        tgt = torch.transpose(tgt,1,2)
        tgt = torch.matmul(ratiostgt,tgt)
        tgt = tgt.squeeze(1)
        return tgt

    def forward(self, spectra:torch.Tensor, 
                precursors: torch.Tensor, 
                sequences):

        pkt_features = self.encode_spectrum(spectra=spectra,precursors=precursors)
        tgt_features = self.encode_peptide(sequences=sequences)

        pkt_features = pkt_features / pkt_features.norm(dim = -1, keepdim = True)
        tgt_features = tgt_features / tgt_features.norm(dim = -1, keepdim = True)

        logit_scale = self.logit_scale.exp()
        logits_per_spec = logit_scale * pkt_features @ tgt_features.t()
        logits_per_tgt = logits_per_spec.t()

        return logits_per_spec, logits_per_tgt

    @property
    def device(self):
        """The current device for the model"""
        return next(self.parameters()).device

