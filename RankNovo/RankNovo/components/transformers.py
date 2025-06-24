"""Base Transformer models for working with mass spectra and peptides"""
import random
import re

import torch


from .encoders import MassEncoder, PeakEncoder, PositionalEncoder, TrainablePositionEncoding, RotaryPositionalEmbeddings
from ..masses import PeptideMass
from .. import utils2 as utils
import numpy as np
import itertools

import pytorch_lightning as pl
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class SpectrumEncoder(torch.nn.Module):
    """A Transformer encoder for input mass spectra.

    Parameters
    ----------
    dim_model : int, optional
        The latent dimensionality to represent peaks in the mass spectrum.
    n_head : int, optional
        The number of attention heads in each layer. ``dim_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the Transformer
        layers of the model.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    peak_encoder : bool, optional
        Use positional encodings m/z values of each peak.
    dim_intensity: int or None, optional
        The number of features to use for encoding peak intensity.
        The remaining (``dim_model - dim_intensity``) are reserved for
        encoding the m/z value.
    """

    def __init__(
        self,
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0,
        peak_encoder=True,
        dim_intensity=None,
    ):
        """Initialize a SpectrumEncoder"""
        super().__init__()

        # self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, dim_model))
        # self.spectrum_matrix = torch.nn.Parameter(torch.randn(dim_model,dim_model))

        # dim_intensity = 128
        self.zeroPeaks_intensity = torch.nn.Parameter(torch.randn(1,1,1))
        self.allPeaks_intensity = torch.nn.Parameter(torch.randn(1,1,1))

        if peak_encoder:
            self.peak_encoder = PeakEncoder(
                dim_model,
                dim_intensity=dim_intensity,
            )
        else:
            self.peak_encoder = torch.nn.Linear(2, dim_model)

        # The Transformer layers:
        layer = torch.nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
        )
        
        # Precursor Encoder
        # self.mass_encoder = MassEncoder(dim_model=256)
        # self.charge_encoder = torch.nn.Embedding(10, dim_model)

    def forward(self, spectra, precursors):
        """The forward pass.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.

        Returns
        -------
        latent : torch.Tensor of shape (n_spectra, n_peaks + 1, dim_model)
            The latent representations for the spectrum and each of its
            peaks.
        mem_mask : torch.Tensor
            The memory mask specifying which elements were padding in X.
        """

        # add percursors into encoder
        # masses = self.mass_encoder(precursors[:, None, [0]])
        # charges = self.charge_encoder(precursors[:, 1].int() - 1)
        # precursors = masses + charges[:, None, :]


        zeroMass = torch.zeros([precursors.shape[0],1,1]).to(self.device) # [n_spectra, 1, 1] mass = 0
        precursorMass = precursors[:, None, [0]] # [n_spectra, 1, 1] mass = precursor

        zeroPeaksIntensities = self.zeroPeaks_intensity.expand(precursors.shape[0],-1,-1) # (n_spectra, 1, 1)
        allPeaksIntensities = self.allPeaks_intensity.expand(precursors.shape[0],-1,-1)
        zeros = torch.cat([zeroMass, zeroPeaksIntensities],dim = 2) # (n_spectra, 1, 2)
        alls = torch.cat([precursorMass, allPeaksIntensities],dim = 2)
        starts = torch.cat([zeros, alls],dim = 1) # (n_spectra, 2, 2)
        spectra = torch.cat([starts, spectra],dim = 1) # (n_spectra, 2 + n_peaks, 2)

        zeros = ~spectra.sum(dim=2).bool() # (n_spectra, n_peaks) 
        mask = zeros
        # mask = [
        #     # add percursors into encoder
        #     # torch.tensor([[False]] * spectra.shape[0]).type_as(zeros),
        #     # torch.tensor([[False]] * spectra.shape[0]).type_as(zeros),
        #     zeros,
        # ]
        # mask = torch.cat(mask, dim=1)
        

        peaks = self.peak_encoder(spectra)

        # precursors = torch.matmul(precursors, self.spectrum_matrix)
        # peaks = torch.concat([precursors, peaks], dim = 1) 

        # Add the spectrum representation to each input:
        # latent_spectra = self.latent_spectrum.expand(peaks.shape[0], -1, -1)
        # peaks = torch.cat([latent_spectra, peaks], dim=1)
        
        return self.transformer_encoder(peaks, src_key_padding_mask=mask), mask

    @property
    def device(self):
        """The current device for the model"""
        return next(self.parameters()).device


class _PeptideTransformer(torch.nn.Module):
    """A transformer base class for peptide sequences.

    Parameters
    ----------
    dim_model : int
        The latent dimensionality to represent the amino acids in a peptide
        sequence.
    pos_encoder : bool
        Use positional encodings for the amino acid sequence.
    residues: Dict or str {"massivekb", "canonical"}, optional
        The amino acid dictionary and their masses. By default this is only
        the 20 canonical amino acids, with cysteine carbamidomethylated. If
        "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    max_charge : int
        The maximum charge to embed.
    """

    def __init__(
        self,
        dim_model,
        pos_encoder,
        residues,
        max_charge,
    ):
        super().__init__()
        self.reverse = True
        self._peptide_mass = PeptideMass(residues=residues)
        self._amino_acids = list(self._peptide_mass.masses.keys()) + ["$"]
        self._idx2aa = {i + 1: aa for i, aa in enumerate(self._amino_acids)} # 1~28
        self._aa2idx = {aa: i for i, aa in self._idx2aa.items()}

        if pos_encoder:
            self.pos_encoder = PositionalEncoder(dim_model)
        else:
            self.pos_encoder = torch.nn.Identity()

        self.charge_encoder = torch.nn.Embedding(max_charge, dim_model)
        self.aa_encoder = torch.nn.Embedding(
            len(self._amino_acids) + 1,
            dim_model,
            padding_idx=0,
        )

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

        if not partial:
            sequence += ["$"]

        tokens = [self._aa2idx[aa] for aa in sequence]
        tokens = torch.tensor(tokens, device=self.device)
        return tokens

    def deMass(self,sequence):

        if not isinstance(sequence, str):
            
            sequence = [self._idx2aa.get(i.item(), "") for i in sequence]
            masses = [self._peptide_mass.masses[aa] for aa in sequence]
            # if(len(sequence) > 1):
            #     print(sequence,masses)
            masses = list(itertools.accumulate(masses))
            masses = torch.tensor(masses, device = self.device)
   
            return masses
        
        sequence = sequence.replace("I", "L")
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)

        if self.reverse:
            sequence = list(reversed(sequence))

        masses = [self._peptide_mass.masses[aa] for aa in sequence]
        masses = list(itertools.accumulate(masses))
        masses.append(0.0)

        masses = torch.tensor(masses, device = self.device)

        return masses
    
    def get_suffix_mass(self,sequence,premass):

        if not isinstance(sequence, str):

            sequence = [self._idx2aa.get(i.item(), "") for i in sequence]
            masses = [self._peptide_mass.masses[aa] for aa in sequence]
            masses = list(itertools.accumulate(masses))
            masses = torch.tensor(masses, device = self.device)
            masses = premass - masses
            # print(sequence,masses)
            return masses
        
        sequence = sequence.replace("I", "L")
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)

        if self.reverse:
            sequence = list(reversed(sequence))

        masses = [self._peptide_mass.masses[aa] for aa in sequence]

        masses = list(itertools.accumulate(masses))
        masses.append(premass)

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
        masstemp = torch.cat([masstemp,torch.tensor([0.0]).to(masstemp.device)])

        return masstemp
    
    def getAminoAcid(self):
        AA_masslist = [self._peptide_mass.masses[self._idx2aa[i]] for i in range(1, len(self._amino_acids))]
        AA_masslist = [0] + AA_masslist
        AA_masslist = torch.tensor(AA_masslist, device = self.device)
        return AA_masslist

    def detokenize(self, tokens):
        """Transform tokens back into a peptide sequence.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_amino_acids,)
            The token for each amino acid in the peptide sequence.

        Returns
        -------
        list of str
            The amino acids in the peptide sequence.
        """
        sequence = [self._idx2aa.get(i.item(), "") for i in tokens]
        if "$" in sequence:
            idx = sequence.index("$")
            sequence = sequence[: idx + 1]

        if self.reverse:
            sequence = list(reversed(sequence))

        return sequence

    @property
    def vocab_size(self):
        """Return the number of amino acids"""
        return len(self._aa2idx)

    @property
    def device(self):
        """The current device for the model"""
        return next(self.parameters()).device


# class PeptideEncoder(_PeptideTransformer):
#     """A transformer encoder for peptide sequences.

#     Parameters
#     ----------
#     dim_model : int
#         The latent dimensionality to represent the amino acids in a peptide
#         sequence.
#     n_head : int, optional
#         The number of attention heads in each layer. ``dim_model`` must be
#         divisible by ``n_head``.
#     dim_feedforward : int, optional
#         The dimensionality of the fully connected layers in the Transformer
#         layers of the model.
#     n_layers : int, optional
#         The number of Transformer layers.
#     dropout : float, optional
#         The dropout probability for all layers.
#     pos_encoder : bool, optional
#         Use positional encodings for the amino acid sequence.
#     residues: Dict or str {"massivekb", "canonical"}, optional
#         The amino acid dictionary and their masses. By default this is only
#         the 20 canonical amino acids, with cysteine carbamidomethylated. If
#         "massivekb", this dictionary will include the modifications found in
#         MassIVE-KB. Additionally, a dictionary can be used to specify a custom
#         collection of amino acids and masses.
#     max_charge : int, optional
#         The maximum charge state for peptide sequences.
#     """

#     def __init__(
#         self,
#         dim_model=128,
#         n_head=8,
#         dim_feedforward=1024,
#         n_layers=1,
#         dropout=0,
#         pos_encoder=True,
#         residues="canonical",
#         max_charge=5,
#     ):
#         """Initialize a PeptideEncoder"""
#         super().__init__(
#             dim_model=dim_model,
#             pos_encoder=pos_encoder,
#             residues=residues,
#             max_charge=max_charge,
#         )

#         # The Transformer layers:
#         layer = torch.nn.TransformerEncoderLayer(
#             d_model=dim_model,
#             nhead=n_head,
#             dim_feedforward=dim_feedforward,
#             batch_first=True,
#             dropout=dropout,
#         )

#         self.transformer_encoder = torch.nn.TransformerEncoder(
#             layer,
#             num_layers=n_layers,
#         )

#     def forward(self, sequences, charges):
#         """Predict the next amino acid for a collection of sequences.

#         Parameters
#         ----------
#         sequences : list of str or list of torch.Tensor of length batch_size
#             The partial peptide sequences for which to predict the next
#             amino acid. Optionally, these may be the token indices instead
#             of a string.
#         charges : torch.Tensor of size (batch_size,)
#             The charge state of the peptide

#         Returns
#         -------
#         latent : torch.Tensor of shape (n_sequences, len_sequence, dim_model)
#             The latent representations for the spectrum and each of its
#             peaks.
#         mem_mask : torch.Tensor
#             The memory mask specifying which elements were padding in X.
#         """
#         sequences = utils.listify(sequences)
#         tokens = [self.tokenize(s) for s in sequences]
#         tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
#         encoded = self.aa_encoder(tokens)

#         # Encode charges
#         charges = self.charge_encoder(charges - 1)[:, None]
#         encoded = torch.cat([charges, encoded], dim=1)

#         # Create mask
#         mask = ~encoded.sum(dim=2).bool()

#         # Add positional encodings
#         encoded = self.pos_encoder(encoded)

#         # Run through the model:
#         latent = self.transformer_encoder(encoded, src_key_padding_mask=mask)
#         return latent, mask


class PeptideDecoder(_PeptideTransformer):
    """A transformer decoder for peptide sequences.

    Parameters
    ----------
    dim_model : int, optional
        The latent dimensionality to represent peaks in the mass spectrum.
    n_head : int, optional
        The number of attention heads in each layer. ``dim_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the Transformer
        layers of the model.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    pos_encoder : bool, optional
        Use positional encodings for the amino acid sequence.
    reverse : bool, optional
        Sequence peptides from c-terminus to n-terminus.
    residues: Dict or str {"massivekb", "canonical"}, optional
        The amino acid dictionary and their masses. By default this is only
        the 20 canonical amino acids, with cysteine carbamidomethylated. If
        "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    """

    def __init__(
        self,
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0,
        pos_encoder=True,
        reverse=True,
        residues="canonical",
        max_charge=5,
    ):
        """Initialize a PeptideDecoder"""
        super().__init__(
            dim_model=dim_model,
            pos_encoder=pos_encoder,
            residues=residues,
            max_charge=max_charge,
        )
        self.reverse = reverse
        
        self.aaDim = dim_model - dim_model // 2
        
        # self.mass_encoder = MassEncoder(dim_model)
        # self.charge_encoder = torch.nn.Embedding(max_charge, dim_model)
        # self.aa_encoder = torch.nn.Embedding(
        #     len(self._amino_acids) + 1,
        #     dim_model,
        #     padding_idx=0,
        # )
        
        # self.mass_fc = torch.nn.Linear(1, dim_model, bias=False)
        
        # precursor encoder
        self.mass_encoder = MassEncoder(dim_model // 2)
        self.charge_encoder = torch.nn.Embedding(max_charge, dim_model // 2)

        # aa encoder
        self.prefixMassEncoder = MassEncoder(dim_model // 4)
        self.suffixMassEncoder = MassEncoder(dim_model // 4)
        self.aa_encoder = torch.nn.Embedding(
            len(self._amino_acids) + 1,
            self.aaDim,
            padding_idx=0,
        )
        
        # look up table
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.final_aa_Encoder = torch.nn.Embedding(
            len(self._amino_acids) + 1,
            self.aaDim,
            padding_idx=0, 
        )
        self.final_mass_encoder = MassEncoder(dim_model // 2)
        self.finalCharMass = torch.nn.Parameter(torch.randn(1))
        finalLinears = []
        xin = dim_model
        for xout in [512,1024,1024]:
            finalLinears.append(torch.nn.Linear(xin,xout))
            finalLinears.append(torch.nn.PReLU())
            xin = xout
        finalLinears.append(torch.nn.Linear(xin, dim_model))
        self.finalLinears = torch.nn.Sequential(*finalLinears)
        
        # self.final = torch.nn.Linear(dim_model, len(self._amino_acids) + 1)
        
        # position
        # self.pos_encoder = TrainablePositionEncoding(dim_model)
        # self.pos_encoder = RotaryPositionalEmbeddings(dim_model)

        layer = torch.nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer_decoder = torch.nn.TransformerDecoder(
            layer,
            num_layers=n_layers,
        )

    def forward(self, sequences, precursors, memory, memory_key_padding_mask):
        """Predict the next amino acid for a collection of sequences.

        Parameters
        ----------
        sequences : list of str or list of torch.Tensor
            The partial peptide sequences for which to predict the next
            amino acid. Optionally, these may be the token indices instead
            of a string.
        precursors : torch.Tensor of size (batch_size, 2)
            The measured precursor mass (axis 0) and charge (axis 1) of each
            tandem mass spectrum
        memory : torch.Tensor of shape (batch_size, n_peaks, dim_model)
            The representations from a ``TransformerEncoder``, such as a
           ``SpectrumEncoder``.
        memory_key_padding_mask : torch.Tensor of shape (batch_size, n_peaks)
            The mask that indicates which elements of ``memory`` are padding.

        Returns
        -------
        scores : torch.Tensor of size (batch_size, len_sequence, n_amino_acids)
            The raw output for the final linear layer. These can be Softmax
            transformed to yield the probability of each amino acid for the
            prediction.
        tokens : torch.Tensor of size (batch_size, len_sequence)
            The input padded tokens.

        """

        if sequences is not None:
            
            # no beam
            tmp = []
            for s in sequences:
                tmp.append(s[0])
            sequences = tmp
            
            sequences = utils.listify(sequences)
            # mass = [self.get_mass(s) for s in sequences]
            # mass = torch.nn.utils.rnn.pad_sequence(mass, batch_first = True)
            preMasses = [self.deMass(s) for s in sequences]
            preMasses = torch.nn.utils.rnn.pad_sequence(preMasses, batch_first = True)
            suffixMasses = [self.get_suffix_mass(sequences[i],precursors[i][0]) for i in range(len(sequences))]
            suffixMasses = torch.nn.utils.rnn.pad_sequence(suffixMasses, batch_first = True)
            tokens = [self.tokenize(s) for s in sequences]
            tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first = True)
        else:
            preMasses = torch.tensor([[]]).to(self.device)
            suffixMasses = torch.tensor([[]]).to(self.device)
            tokens = torch.tensor([[]]).to(self.device)

        # precursors
        masses = self.mass_encoder(precursors[:, None, [0]])
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        precursors = masses + charges[:, None, :]
        
        # aa
        tgt = self.aa_encoder(tokens.to(torch.long))
        tgt_key_padding_mask = tgt.sum(axis=2) == 0
        
        # mass = mass.unsqueeze(2)
        # mass = self.final_mass_encoder(mass)
        # tgt = tgt + mass
        
        # mass
        preAndSufPrecursors = torch.tensor([[0]]).to(self.device)
        preAndSufPrecursors = self.prefixMassEncoder(preAndSufPrecursors)
        preAndSufPrecursors = preAndSufPrecursors.repeat(precursors.shape[0],1)
        preAndSufPrecursors = preAndSufPrecursors.unsqueeze(1)
        precursors = torch.cat([precursors,preAndSufPrecursors,preAndSufPrecursors],dim=2)

        preMasses = preMasses.unsqueeze(2)
        preMasses = self.prefixMassEncoder(preMasses)
        suffixMasses = suffixMasses.unsqueeze(2)
        suffixMasses = self.suffixMassEncoder(suffixMasses)
        allMasses = torch.concat([preMasses,suffixMasses],dim=2)
        tgtTemp = torch.concat([tgt,allMasses], dim=2)
        
        # Feed through model:
        if sequences is None:
            tgt = precursors
        else:
            tgt = torch.cat([precursors, tgtTemp], dim=1)
        # precursors_mask = torch.zeros(tgt.size(0), 1, device=tgt.device, dtype=torch.bool)
        # tgt_key_padding_mask = torch.cat((precursors_mask, tgt_key_padding_mask), dim=1)
        tgt_key_padding_mask = tgt.sum(axis=2) == 0 
        tgt = self.pos_encoder(tgt)
        tgt_mask = generate_tgt_mask(tgt.shape[1]).type_as(precursors)
        preds = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask.to(self.device),
        )
        
        # look up table
        aa_masses = self.getAminoAcid()
        aa_masses = torch.concat([aa_masses, self.finalCharMass], dim=0).unsqueeze(1) 
        aa_masses = self.final_mass_encoder(aa_masses)
        aa_idx = torch.arange(0, len(self._amino_acids) + 1).to(torch.long).to(self.device) 
        aa_idx = self.final_aa_Encoder(aa_idx)
        final_martix = torch.concat([aa_masses, aa_idx], dim = -1)
        final_martix = self.finalLinears(final_martix)
        preds = self.logit_scale * preds @ final_martix.t()
        
        # preds = self.final(preds)

        return preds, tokens 

        return torch.softmax(preds, dim=2), tokens

        # return torch.softmax(self.final(preds),dim=2), tokens


class CrossEncoder(_PeptideTransformer):

    def __init__(
        self,
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0,
        pos_encoder=True,
        residues="canonical",
        max_charge=5,
        dim_intensity=None,
    ):
        """Initialize a PeptideEncoder"""
        super().__init__(
            dim_model=dim_model,
            pos_encoder=pos_encoder,
            residues=residues,
            max_charge=max_charge,
        )
        
        
        self.zeroPeaks_intensity = torch.nn.Parameter(torch.randn(1,1,1))
        self.allPeaks_intensity = torch.nn.Parameter(torch.randn(1,1,1))
        self.peak_encoder = PeakEncoder(
            dim_model,
            dim_intensity=None,
        )
        
        self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, dim_model))
        
        self.token_type_embeddings = torch.nn.Embedding(2, dim_model)
        
        # MassEncoder for prefix and suffix mass
        self.aminoEmbedDim = dim_model // 2
        self.prefixMassEncoder = MassEncoder(dim_model // 4)
        self.suffixMassEncoder = MassEncoder(dim_model // 4)
        self.massEncoder = MassEncoder(dim_model)
        self.aa_encoder = torch.nn.Embedding(
            len(self._amino_acids) + 1,
            self.aminoEmbedDim,
            padding_idx=0,
        )

        # The Transformer layers:
        layer = torch.nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
        )

    def forward(self, spectra, sequences, precursors):
        
        # spectrum
        zeroMass = torch.zeros([precursors.shape[0],1,1]).to(self.device) # [n_spectra, 1, 1] mass = 0
        precursorMass = precursors[:, None, [0]] # [n_spectra, 1, 1] mass = precursor
        zeroPeaksIntensities = self.zeroPeaks_intensity.expand(precursors.shape[0],-1,-1) # (n_spectra, 1, 1)
        allPeaksIntensities = self.allPeaks_intensity.expand(precursors.shape[0],-1,-1)
        zeros = torch.cat([zeroMass, zeroPeaksIntensities],dim = 2) # (n_spectra, 1, 2)
        alls = torch.cat([precursorMass, allPeaksIntensities],dim = 2)
        starts = torch.cat([zeros, alls],dim = 1) # (n_spectra, 2, 2)
        spectra = torch.cat([starts, spectra],dim = 1) # (n_spectra, 2 + n_peaks, 2)
        
        mask1 = ~spectra.sum(dim=2).bool() # (n_spectra, n_peaks) 
        peaks = self.peak_encoder(spectra)

        spectra_type_ids = torch.zeros_like(peaks[:, :, 0], dtype=torch.long) 
        spectra_type_embeddings = self.token_type_embeddings(spectra_type_ids)
        peaks = peaks + spectra_type_embeddings
        
        # Process each batch of sequences
        encoded_sequences = []
        sequence_batch_indices = []
        all_mass = []
        preMasses = []
        suffixMasses = []
        for batch_seqs in sequences:
            batch_encoded = []
            for seq in batch_seqs: 
                all_mass.append(self._peptide_mass.mass(seq))
                preMasses.append(self.deMass(seq))
                suffixMasses.append(self.get_suffix_mass(seq, all_mass[-1]))
                
                token = self.tokenize(seq)
                sequence_encoded = self.aa_encoder(token)
                batch_encoded.append(sequence_encoded)
            encoded_sequences.extend(batch_encoded)
            sequence_batch_indices.append(len(batch_encoded))  # Store the count of sequences for each batch
            # print(len(batch_encoded))
            
        preMasses = torch.nn.utils.rnn.pad_sequence(preMasses, batch_first = True)
        suffixMasses = torch.nn.utils.rnn.pad_sequence(suffixMasses, batch_first = True)
        preMasses = preMasses.unsqueeze(2)
        preMasses = self.prefixMassEncoder(preMasses)
        suffixMasses = suffixMasses.unsqueeze(2)
        suffixMasses = self.suffixMassEncoder(suffixMasses)
        Masses = torch.concat([preMasses, suffixMasses], dim=2)
        
        padded_sequences = torch.nn.utils.rnn.pad_sequence(encoded_sequences, batch_first=True)
        padded_sequences = torch.concat([padded_sequences, Masses],dim=2)
        padded_masks = ~padded_sequences.sum(dim=2).bool()
        
        # position encode
        padded_sequences = self.pos_encoder(padded_sequences)

        # Combine spectra and sequences
        combined_input = []
        combined_mask = []
        index = 0
        for i, num_sequences in enumerate(sequence_batch_indices):
            for j in range(num_sequences):
                sequence = padded_sequences[index:index+1]
                sequences_type_ids = torch.ones_like(sequence[:, :, 0], dtype=torch.long)
                sequences_type_embeddings = self.token_type_embeddings(sequences_type_ids)
                sequence = sequence + sequences_type_embeddings

                combined = torch.cat([peaks[i:i+1], sequence], dim=1)
                combined_input.append(combined)

                mask = torch.cat([mask1[i:i+1], padded_masks[index:index+1]], dim=1)
                combined_mask.append(mask)

                index += 1

        combined_input = torch.cat(combined_input, dim=0)
        combined_mask = torch.cat(combined_mask, dim=0)
        
        # Add the spectrum representation to each input:
        latent_spectra = self.latent_spectrum.expand(combined_input.shape[0], -1, -1)
        combined_input = torch.cat([latent_spectra, combined_input], dim=1)
        
        combined_mask = [
            torch.tensor([[False]] * combined_input.shape[0]).type_as(combined_mask),
            combined_mask,
        ]
        combined_mask = torch.cat(combined_mask, dim=1)

        # Run through the model:
        latent = self.transformer_encoder(combined_input, src_key_padding_mask=combined_mask)
        
        return latent, mask, sequence_batch_indices






def generate_tgt_mask(sz):
    """Generate a square mask for the sequence. The masked positions
    are filled with float('-inf'). Unmasked positions are filled with
    float(0.0).

    This function is a slight modification of the version in the PyTorch
    repository.

    Parameters
    ----------
    sz : int
        The length of the target sequence.
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask

def reverse_sequence_with_probability(sequence, p):
    """
    Reverse the given sequence with a probability p.

    Parameters:
    - sequence: a list of elements to be potentially reversed.
    - p: a float representing the probability of reversing the sequence.
         It should be a value between 0 and 1.
    
    Returns:
    The original sequence or the reversed sequence based on the probability p.
    """
    assert 0 <= p <= 1, "Probability p must be between 0 and 1"
    
    # Decide whether to reverse the sequence based on probability p
    if random.random() < p:
        return sequence[::-1]  # Reverse the sequence
    else:
        return sequence  # Return the sequence as is