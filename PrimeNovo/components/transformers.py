"""Base Transformer models for working with mass spectra and peptides"""
import re
import copy
import torch

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
            padding_idx=-1, ## to be checked weather need the padding
        )
    def get_pad_idx(self):
        #return the idx number for padding token, which is not in dictonary
        return -1 

    def get_blank_idx (self):
        #return the idx in dic for ctc blank token 
        return self._aa2idx["_"]
    def get_blank_sym (self):
        #return the blank symbol used in dic
        return "_"
    def get_symbols(self):
        symbols = []
        for i in range(len(self._aa2idx)):
            symbols.append(self._idx2aa[i])
        return symbols
    def tokenize(self, sequence, partial=False):
        """Transform a peptide sequence into tokens

        Parameters
        ----------
        sequence : str
            A peptide sequence.

        Returns
        -------
        torch.Tensor
            The token for each amino acid in the peptide sequence. which is unblanced matrix, not rectangle 
        """
        if not isinstance(sequence, str):
            return sequence  # Assume it is already tokenized.

        sequence = sequence.replace("I", "L")
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)
        if self.reverse:
            sequence = list(reversed(sequence))

        #if not partial:
            #sequence += ["$"]

        tokens = [self._aa2idx[aa] for aa in sequence]
        tokens = torch.tensor(tokens, device=self.device)
        return tokens
    def remove_repentance(self, index_list) :
        """
        Eliminate repeated index in "a" list. e.g., [1, 1, 2, 2, 3] --> [1, 2, 3]
        """
        return [a for a, b in zip(index_list, index_list[1:] + [not index_list[-1]]) if a != b]
        
    def ctc_post_processing(self, sentence_index):
        # setence_index: list of index of a peptide 
        sentence_index = self.remove_repentance(sentence_index)
        sentence_index = list(filter((self.get_blank_idx()).__ne__, sentence_index))
        #sentence_index = list(filter((self.dictionary.pad()).__ne__, sentence_index))
        return sentence_index

    def detokenize_truth(self, tokens, is_beam=False):
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
        #sequence = [self._idx2aa.get(i.item(), "") for i in tokens]
        if is_beam == False:
            sequence = [i.item() for i in tokens]
        else:
            sequence = tokens
        sequence = list(filter((self.get_pad_idx()).__ne__, sequence))

        sequence = [self._idx2aa[i] for i in sequence] # list[str], 
        
        '''
        if "$" in sequence:
            idx = sequence.index("$")
            sequence = sequence[: idx + 1]
        '''
        
        
        if self.reverse:
            sequence = list(reversed(sequence))

        return sequence  
    
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
        #sequence = [self._idx2aa.get(i.item(), "") for i in tokens]
        sequence = [i.item() for i in tokens]
        sequence = self.ctc_post_processing(sequence)

        sequence = [self._idx2aa[i] for i in sequence] # list["str"], 
        
        '''
        if "$" in sequence:
            idx = sequence.index("$")
            sequence = sequence[: idx + 1]
        '''
        
        
        if self.reverse:
            sequence = list(reversed(sequence))

        return sequence  # list 

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
        tokens = [self.tokenize(s) for s in sequences]
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
        max_pep_len = 100
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


        self.mass_mapping = torch.tensor(tem_list).to(self.device) # vocab_size 

        # Additional model components
        self.mass_encoder = MassEncoder(dim_model)
        layer = torch.nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = dropout
        self.layers = _get_clones(layer, n_layers)
        self.layer_ln = torch.nn.Linear (dim_model * 2 , dim_model)

        self.final = torch.nn.Linear(dim_model, len(self._amino_acids))
    def demass(self, tokens_pred):
        #tokens_pred  = (bz, seq_len)
        
        token_onehot = torch.nn.functional.one_hot(tokens_pred, num_classes=self.vocab_size).to(self.device) # bz, seq_len, vocab_size
        mass_mapped  = token_onehot.to(self.mass_mapping.dtype) @ self.mass_mapping.to(self.device)  #bz, seq_len
        #mass_mapped = mass_mapped[:, :, None ] # bz, seq_len, 1 
        
        return mass_mapped  #bz, seq_len

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
            tokens = [self.tokenize(s) for s in sequences]
            tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value = self.get_pad_idx())
        else:
            tokens = torch.tensor([[]]).to(self.device)
        

        # Prepare mass and charge
        masses = self.mass_encoder(precursors[:, None, [0]])  #(bz, 1, dim)
        charges = self.charge_encoder(precursors[:, 1].int() - 1)  #(bz, dim)
        precursors = masses + charges[:, None, :] # bz, 1, dim

        # Feed through model:
        tgt = precursors.repeat(1, self.max_pep_len, 1)    # b_z, max_len, dim
        '''
        if sequences is None:
            tgt = precursors
        else:
            tgt = torch.cat([precursors, self.aa_encoder(tokens)], dim=1) # to be changed, no longer need any encoder for peptite in the input 
        '''
        tgt_key_padding_mask = tgt.sum(axis=2) == 0
        tgt = self.pos_encoder(tgt)
        #tgt_mask = generate_tgt_mask(tgt.shape[1]).type_as(precursors)
        '''
        preds = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            #tgt_mask=tgt_mask,
            #tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask.to(self.device),
        ) #bz, token_len, dim
        '''
        
        output = tgt

        output_list = []

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=None,
                         
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            preds = self.final(output) #bz, token_len, dic_size
            output_list.append(preds) 
            #pred_tokens = torch.argmax(preds, axis=2) #bz, token_len
            #encoded_tokens = self.aa_encoder(pred_tokens) #bz, token_len, dim_model
            #-------------not useful-------
            #cum_mass = self.demass(pred_tokens) #bz, token_len
            #cum_mass = cum_mass[:, :, None] #bz, token_len, 1
            #encoded_mass = self.mass_ln(cum_mass) #bz, token_len, dim_model 
            #aa_encoded = encoded_mass + encoded_tokens  #bz, token_len, dim_model
            #--------------------------------
            #total_encod = torch.cat([output, encoded_tokens], dim=2) # bz, token_len ,dim_model * 2 
            #output = self.layer_ln(total_encod)
            
            #output = torch.nn.functional.dropout(output, p=self.dropout)
        
            


        






        
        return output_list[-1], tokens, output_list

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])


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
