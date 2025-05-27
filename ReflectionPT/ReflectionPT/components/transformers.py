"""reflection"""
import re

import torch
import random

from .encoders import MassEncoder, PeakEncoder, PositionalEncoder
from ..masses import PeptideMass
from .. import utils



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

        self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, dim_model))

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
        self.reverse = False
        self._peptide_mass = PeptideMass(residues=residues)
        self._amino_acids = list(self._peptide_mass.masses.keys()) + ["<back>","$"]
        self._idx2aa = {i + 1: aa for i, aa in enumerate(self._amino_acids)} #1 for padding token, at index 0 
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

    def tokenize(self, sequence, partial=False, use=1):
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
            return sequence, 0  # Assume it is already tokenized.

        sequence = sequence.replace("I", "L")
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)
        if self.reverse:
            sequence = list(reversed(sequence))

        if not partial:
            sequence += ["$"]

        tokens = [self._aa2idx[aa] for aa in sequence] 
        


        #add xx: insert errors:
        #default: 40% sequence, randomly inserting 1-3 errors 
        error_labels = [0] * len(tokens)
        if  random.random() <= 0.98 and use == 1: #decide weather this sequence needed a error to be insert
            k = random.randint(1, 10) #k = number of errors 
            #insert_positions = sorted(random.sample(range(len(tokens)), k))

            #non-adjcent implementation
            insert_positions = []
            available_positions = set(range(len(tokens)))

            while len(insert_positions) < k and available_positions:
                pos = random.choice(list(available_positions))
                insert_positions.append(pos)
                # Remove adjacent positions to avoid placing tokens next to each other
                available_positions.discard(pos)
                available_positions.discard(pos - 1)
                available_positions.discard(pos + 1)

            insert_positions.sort()

            offset = 0
            for pos in insert_positions:

                #---future error sampling
                start_idx = pos + offset + 1
                end_idx = start_idx + 4
                future_tokens = tokens[start_idx:end_idx]
                if future_tokens and random.random() <= 0.85:
                    error_token = random.choice(future_tokens)
                else:
                    error_token = random.randint(1, len(self._amino_acids) - 2)





                #error_token = random.randint(1, len(self._amino_acids)-2) #might need fix as some tokens can not be in the middle, but might be ok since will be corrected by model first  
                tokens.insert(pos + offset, error_token)
                tokens.insert(pos + offset + 1, self._aa2idx["<back>"])
                error_labels.insert(pos + offset, 1)
                error_labels.insert(pos + offset + 1, 0)
                offset += 2  # Increment offset by 2 for each insertion
        #-------------------

        tokens = torch.tensor(tokens, device=self.device)
        error_labels = torch.tensor(error_labels, device=self.device)
        return tokens, error_labels #error labels having 1 at position of inserted error

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
        sequence = []
        sequence_original = []
        hit_stop = False
        for i in tokens:
            aa = self._idx2aa.get(i.item(), "")
            sequence_original.append(aa)
            if aa == "$":
                hit_stop=True
            if aa == "<back>":
                if sequence and not hit_stop:
                    sequence.pop()
                continue 
            sequence.append(aa)
        #sequence = [self._idx2aa.get(i.item(), "") for i in tokens]
        if "$" in sequence:
            idx = sequence.index("$")
            sequence = sequence[: idx + 1]
        if "$" in sequence_original:
            idx = sequence_original.index("$")
            sequence_original = sequence_original[:idx+1]
        assert "<back>" not in sequence 
        
        if sequence == []:
            sequence = ["$"]

        if self.reverse:
            sequence = list(reversed(sequence))

        return sequence, sequence_original

    @property
    def vocab_size(self):
        """Return the number of amino acids"""
        return len(self._aa2idx)

    @property
    def device(self):
        """The current device for the model"""
        return next(self.parameters()).device


class PeptideEncoder(_PeptideTransformer):
    """A transformer encoder for peptide sequences.

    Parameters
    ----------
    dim_model : int
        The latent dimensionality to represent the amino acids in a peptide
        sequence.
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
    residues: Dict or str {"massivekb", "canonical"}, optional
        The amino acid dictionary and their masses. By default this is only
        the 20 canonical amino acids, with cysteine carbamidomethylated. If
        "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    max_charge : int, optional
        The maximum charge state for peptide sequences.
    """

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
    ):
        """Initialize a PeptideEncoder"""
        super().__init__(
            dim_model=dim_model,
            pos_encoder=pos_encoder,
            residues=residues,
            max_charge=max_charge,
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

    def forward(self, sequences, charges):
        """Predict the next amino acid for a collection of sequences.

        Parameters
        ----------
        sequences : list of str or list of torch.Tensor of length batch_size
            The partial peptide sequences for which to predict the next
            amino acid. Optionally, these may be the token indices instead
            of a string.
        charges : torch.Tensor of size (batch_size,)
            The charge state of the peptide

        Returns
        -------
        latent : torch.Tensor of shape (n_sequences, len_sequence, dim_model)
            The latent representations for the spectrum and each of its
            peaks.
        mem_mask : torch.Tensor
            The memory mask specifying which elements were padding in X.
        """
        sequences = utils.listify(sequences)
        tokens, error_labels = map(list, zip(*[self.tokenize(s) for s in sequences]))
        error_labels = torch.nn.utils.rnn.pad_sequence(error_labels, batch_first=True, padding_value=0)

        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        encoded = self.aa_encoder(tokens)

        # Encode charges
        charges = self.charge_encoder(charges - 1)[:, None]
        encoded = torch.cat([charges, encoded], dim=1)

        # Create mask
        mask = ~encoded.sum(dim=2).bool()

        # Add positional encodings
        encoded = self.pos_encoder(encoded)

        # Run through the model:
        latent = self.transformer_encoder(encoded, src_key_padding_mask=mask)
        return latent, mask


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

        # Additional model components
        self.mass_encoder = MassEncoder(dim_model)
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

        self.final = torch.nn.Linear(dim_model, len(self._amino_acids) + 1)

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
        # Prepare sequences
        if sequences is not None:
            sequences = utils.listify(sequences)
            batch = [self.tokenize(s) for s in sequences]
            
            tokens, error_labels = zip(*batch)
            #print("error label", error_labels)
            #tokens, error_labels = map(list, zip(*[self.tokenize(s) for s in sequences]))
            tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
            if not isinstance(error_labels[0], int):
                error_labels = torch.nn.utils.rnn.pad_sequence(error_labels, batch_first=True, padding_value=0)
        else:
            tokens = torch.tensor([[]]).to(self.device)
            error_labels = torch.tensor([[]]).to(self.device) 

        # Prepare mass and charge
        masses = self.mass_encoder(precursors[:, None, [0]])
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        precursors = masses + charges[:, None, :]

        # Feed through model:
        if sequences is None:
            tgt = precursors
        else:
            tgt = torch.cat([precursors, self.aa_encoder(tokens)], dim=1)

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
        return self.final(preds), tokens, error_labels


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
