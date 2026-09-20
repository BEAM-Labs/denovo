"""Transformer components for PTM-aware de novo peptide sequencing with CTC head."""
import re
import copy
import torch
import numpy as np
from .. import utils
from .encoders import MassEncoder, PeakEncoder, PositionalEncoder
from .qwen import Qwen3Encoder, Qwen3Decoder
from loguru import logger

class PeptideMass_PTM:
    """Calculator for peptide masses including post-translational modifications."""

    # Constants
    hydrogen = 1.007825035
    oxygen = 15.99491463
    h2o = 2 * hydrogen + oxygen
    proton = 1.00727646688

    def __init__(self, residues="canonical"):
        """Initialize the PeptideMass object"""
        self.masses = residues

    def __len__(self):
        """Return the length of the residue dictionary"""
        return len(self.masses)

    def mass(self, seq, charge=None):
        """Calculate a peptide's mass or m/z.

        Parameters
        ----------
        seq : list or str
            The peptide sequence, using tokens defined in ``self.residues``.
        charge : int, optional
            The charge used to compute m/z. Otherwise the neutral peptide mass
            is calculated

        Returns
        -------
        float
            The computed mass or m/z.
        """
        if isinstance(seq, str):
            seq = re.split(r"(?<=.)(?=[A-Z(])", seq)

        calc_mass = sum([self.masses[aa] for aa in seq]) + self.h2o
        if charge is not None:
            calc_mass = (calc_mass / charge) + self.proton

        return calc_mass



class SpectrumEncoder(torch.nn.Module):
    """A Qwen3-based Transformer encoder for input mass spectra.

    Parameters
    ----------
    dim_model : int, optional
        The latent dimensionality to represent peaks in the mass spectrum.
    n_head : int, optional
        The number of attention heads in each layer.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    peak_encoder : bool, optional
        Use positional encodings m/z values of each peak.
    dim_intensity: int or None, optional
        The number of features to use for encoding peak intensity.
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
        config=None,
    ):
        """Initialize a SpectrumEncoder"""
        super().__init__()

        self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, dim_model))

        if peak_encoder:
            self.peak_encoder = PeakEncoder(
                dim_model,
                dim_intensity=dim_intensity,
            )
        else:
            self.peak_encoder = torch.nn.Linear(2, dim_model)

        self.config = config

        self.transformer_encoder = Qwen3Encoder(config=config)

    def forward(self, spectra):
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
        zeros = ~spectra.sum(dim=2).bool()
        mask = [
            torch.tensor([[False]] * spectra.shape[0]).type_as(zeros),
            zeros,
        ]
        mask = torch.cat(mask, dim=1)
        peaks = self.peak_encoder(spectra)

        # Add the spectrum representation to each input:
        latent_spectra = self.latent_spectrum.expand(peaks.shape[0], -1, -1)

        peaks = torch.cat([latent_spectra, peaks], dim=1)

        return self.transformer_encoder(
            inputs_embeds=peaks,
            attention_mask=mask
            ), mask

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
    residues: Dict or str
        The amino acid dictionary and their masses.
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
        self.reverse = False
        self._peptide_mass = PeptideMass_PTM(residues=residues)
        self._amino_acids = list(self._peptide_mass.masses.keys()) + ["_"]
        self._idx2aa = {i : aa for i, aa in enumerate(self._amino_acids)}
        self._aa2idx = {aa: i for i, aa in self._idx2aa.items()}

        if pos_encoder:
            self.pos_encoder = PositionalEncoder(dim_model)
        else:
            self.pos_encoder = torch.nn.Identity()

        self.charge_encoder = torch.nn.Embedding(max_charge, dim_model)
        self.aa_encoder = torch.nn.Embedding(
            len(self._amino_acids),
            dim_model,
            padding_idx=-1,
        )

    def get_pad_idx(self):
        """Return the idx number for padding token."""
        return -1

    def get_blank_idx(self):
        """Return the idx in dictionary for CTC blank token."""
        return self._aa2idx["_"]

    def get_blank_sym(self):
        """Return the blank symbol used in dictionary."""
        return "_"

    def get_symbols(self):
        symbols = []
        for i in range(len(self._aa2idx)):
            symbols.append(self._idx2aa[i])
        return symbols

    def encode(self, sequences):
        return torch.nn.utils.rnn.pad_sequence(
            [self.tokenize(sequence) for sequence in sequences],
            batch_first=True,
            padding_value=-1
            )

    def tokenize(self, sequence, partial=False):
        """Transform a peptide sequence into tokens.

        Parameters
        ----------
        sequence : str
            A peptide sequence.

        Returns
        -------
        torch.Tensor
            The token for each amino acid in the peptide sequence.
        """
        try:
            if not isinstance(sequence, str):
                return sequence  # Assume it is already tokenized.

            sequence1 = sequence.replace("I", "L")
            sequence = re.split(r"(?<=.)(?=[A-Z(])", sequence1)
            if self.reverse:
                sequence = list(reversed(sequence))

            tokens = [self._aa2idx[aa] for aa in sequence]
            tokens = torch.tensor(tokens, device=self.device)
            return tokens
        except Exception as e:
            logger.error("Error happened in tokenize, returning empty tensor. Info", e)
            return torch.tensor([], device=self.device)


    def remove_repentance(self, index_list):
        """
        Eliminate repeated index in a list. e.g., [1, 1, 2, 2, 3] --> [1, 2, 3]
        """
        return [a for a, b in zip(index_list, index_list[1:] + [not index_list[-1]]) if a != b]

    def remove_repentance_fast(self, index_list):
        """
        Eliminate repeated indices in a list more efficiently.
        e.g., [1, 1, 2, 2, 3] --> [1, 2, 3]
        """
        if not index_list:
            return []

        if isinstance(index_list, np.ndarray):
            mask = np.append(index_list[:-1] != index_list[1:], True)
            return index_list[mask].tolist()

        result = [index_list[0]]
        for item in index_list[1:]:
            if item != result[-1]:
                result.append(item)

        return result

    def ctc_post_processing(self, sentence_index):
        """Apply CTC post-processing: merge repetitions then remove blank tokens."""
        sentence_index = self.remove_repentance(sentence_index)
        sentence_index = list(filter((self.get_blank_idx()).__ne__, sentence_index))
        return sentence_index

    def ctc_post_processing_fast(self, sentence_index):
        """
        Apply CTC post-processing to remove repetitions and blank tokens (optimized).
        """
        is_numpy = isinstance(sentence_index, np.ndarray)
        blank_idx = self.get_blank_idx()

        if is_numpy:
            if len(sentence_index) > 0:
                mask = np.append(sentence_index[:-1] != sentence_index[1:], True)
                sentence_index = sentence_index[mask]
            sentence_index = sentence_index[sentence_index != blank_idx]
            return sentence_index.tolist()
        else:
            sentence_index = self.remove_repentance_fast(sentence_index)
            sentence_index = [idx for idx in sentence_index if idx != blank_idx]
            return sentence_index

    def detokenize_truth(self, tokens, is_beam=False):
        """Transform tokens back into a peptide sequence (no CTC post-processing).

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_amino_acids,)
            The token for each amino acid in the peptide sequence.

        Returns
        -------
        list of str
            The amino acids in the peptide sequence.
        """
        if not is_beam:
            sequence = [i.item() for i in tokens]
        else:
            sequence = tokens
        sequence = list(filter((self.get_pad_idx()).__ne__, sequence))
        sequence = [self._idx2aa[i] for i in sequence]

        if self.reverse:
            sequence = list(reversed(sequence))

        return sequence

    def detokenize_truth_fast(self, tokens, is_beam=False):
        """Transform tokens back into a peptide sequence (optimized, no CTC)."""
        if not is_beam:
            sequence = tokens.cpu().numpy() if tokens.is_cuda else tokens.numpy()
        else:
            sequence = tokens

        pad_idx = self.get_pad_idx()
        mask = sequence != pad_idx
        sequence = sequence[mask]

        idx2aa_list = [self._idx2aa[i] for i in range(max(self._idx2aa.keys()) + 1)]
        sequence = [idx2aa_list[i] for i in sequence]

        if self.reverse:
            sequence = list(reversed(sequence))

        return sequence

    def detokenize(self, tokens):
        """Transform tokens back into a peptide sequence with CTC post-processing.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_amino_acids,)
            The token for each amino acid in the peptide sequence.

        Returns
        -------
        list of str
            The amino acids in the peptide sequence.
        """
        sequence = [i.item() for i in tokens]
        sequence = self.ctc_post_processing(sequence)

        sequence = [self._idx2aa[i] for i in sequence]

        if self.reverse:
            sequence = list(reversed(sequence))

        return sequence

    def detokenize_fast(self, tokens):
        """Transform tokens back into a peptide sequence (optimized, with CTC)."""
        sequence = tokens.cpu().numpy() if tokens.is_cuda else tokens.numpy()
        sequence = self.ctc_post_processing_fast(sequence)

        idx2aa_list = [self._idx2aa[i] for i in range(max(self._idx2aa.keys()) + 1)]
        sequence = [idx2aa_list[i] for i in sequence]

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


class PeptideEncoder(_PeptideTransformer):
    """A transformer encoder for peptide sequences."""

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
        reverse = True
    ):
        """Initialize a PeptideEncoder"""
        super().__init__(
            dim_model=dim_model,
            pos_encoder=pos_encoder,
            residues=residues,
            max_charge=max_charge,
        )
        self.reverse = reverse

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
        self.mass_encoder = MassEncoder(dim_model)

    def forward(self, sequences, precursors):
        """Predict the next amino acid for a collection of sequences."""
        sequences = utils.listify(sequences)
        tokens = [self.tokenize(s) for s in sequences]
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value = len(self._amino_acids))
        encoded = self.aa_encoder(tokens)

        masses = self.mass_encoder(precursors[:, None, [0]])
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        precursors =  masses + charges[:, None, :]
        encoded = torch.cat([precursors, encoded], dim=1)

        mask = ~encoded.sum(dim=2).bool()
        encoded = self.pos_encoder(encoded)

        latent = self.transformer_encoder(encoded, src_key_padding_mask=mask)
        return latent, mask


class PeptideDecoder(_PeptideTransformer):
    """A Qwen3-based transformer decoder for peptide sequences with CTC output."""

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
        max_pep_len = 100,
        config=None,
    ):
        """Initialize a PeptideDecoder"""
        super().__init__(
            dim_model=dim_model,
            pos_encoder=pos_encoder,
            residues=residues,
            max_charge=max_charge,
        )

        self.mass_ln = torch.nn.Linear(1, dim_model)
        self.reverse = reverse
        self.max_pep_len = max_pep_len
        tem_list = []
        for i in range(self.vocab_size):
            if i != self.get_blank_idx():
                a = self._idx2aa[i]
                tem_list.append(self._peptide_mass.masses[a])
            else:
                tem_list.append(0)

        self.mass_mapping = torch.tensor(tem_list).to(self.device)

        # Additional model components
        self.mass_encoder = MassEncoder(dim_model)

        self.config = config
        self.model = Qwen3Decoder(config=config)

        self.dropout = dropout

        self.final = torch.nn.Linear(dim_model, len(self._amino_acids))


    def demass(self, tokens_pred):
        """Convert predicted token indices to their corresponding masses."""
        token_onehot = torch.nn.functional.one_hot(tokens_pred, num_classes=self.vocab_size).to(self.device)
        mass_mapped  = token_onehot.to(self.mass_mapping.dtype) @ self.mass_mapping.to(self.device)
        return mass_mapped

    def forward(self, sequences, precursors, memory, memory_key_padding_mask):
        """Predict the next amino acid for a collection of sequences.

        Parameters
        ----------
        sequences : list of str or list of torch.Tensor
            The partial peptide sequences for which to predict the next
            amino acid.
        precursors : torch.Tensor of size (batch_size, 2)
            The measured precursor mass (axis 0) and charge (axis 1) of each
            tandem mass spectrum.
        memory : torch.Tensor of shape (batch_size, n_peaks, dim_model)
            The representations from the SpectrumEncoder.
        memory_key_padding_mask : torch.Tensor of shape (batch_size, n_peaks)
            The mask that indicates which elements of ``memory`` are padding.

        Returns
        -------
        scores : torch.Tensor of size (batch_size, len_sequence, n_amino_acids)
            The raw output for the final linear layer.
        tokens : torch.Tensor of size (batch_size, len_sequence)
            The input padded tokens.
        """
        # Prepare sequences
        if sequences is not None:
            sequences = utils.listify(sequences)
            tokens = [self.tokenize(s) for s in sequences]
            tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value = self.get_pad_idx())
        else:
            tokens = torch.tensor([[]]).to(self.device)

        # Prepare mass and charge
        masses = self.mass_encoder(precursors[:, None, [0]])
        charges = self.charge_encoder(precursors[:, 1].int() - 1)

        precursors =  masses + charges[:, None, :]

        # Feed through model:
        tgt = precursors.repeat(1, self.max_pep_len, 1)

        tgt_key_padding_mask = tgt.sum(axis=2) == 0

        output = tgt

        output_list = []

        output, all_logits, all_self_attn, all_cross_attn = self.model(
            memory=memory,
            memory_mask=memory_key_padding_mask,
            inputs_embeds=output,
            attention_mask=tgt_key_padding_mask,
        )
        for logits in all_logits:
            preds = self.final(logits)
            output_list.append(preds)

        return output_list[-1], tokens, output_list, output

def _get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def mask_illegal_tokens(preds, illegal_idx):
    """
    Mask illegal token indices by setting their prediction logits to negative infinity.

    Args:
        preds: tensor, shape [batch_size, token_len, dic_size]
        illegal_idx: list of token indices to mask out

    Returns:
        Masked prediction tensor.
    """
    mask = torch.zeros_like(preds)
    mask[:, :, illegal_idx] = float('-inf')
    masked_preds = preds + mask
    return masked_preds


def generate_tgt_mask(sz):
    """Generate a square causal mask for the sequence.

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
