import itertools
import math
import pdb
import time
import signal
import os
from typing import Sequence, Tuple, List, Union
import pickle
import re
from spectrum_utils import fragment_annotation as fa, proforma, utils
import shutil
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from pathlib import Path
from .constants import proteinseq_toks
import numpy as np

RawMSA = Sequence[Tuple[float, str]]

class MassAgent():
    def __init__(self, gap_penalty=51.4238, blur=0.1, blur_max_expand=10, blur_timeout=2, precursor_tol=0.1):
        self.gap_penalty = gap_penalty
        self.blur = blur
        self.blur_max_expand = blur_max_expand
        self.blur_timeout = blur_timeout # 5s timeout
        self.precursor_tol = precursor_tol / gap_penalty

        self.ptm_to_mark = {
            "C+57.021": "0",
            "M+15.995": "1",
            "N+0.984": "2",
            "Q+0.984": "3",
            "+42.011": "4",
            "+43.006-17.027": "5",
            "+43.006": "6",
            "-17.027": "7"
        }
        self.mark_to_ptm = {}
        for k, v in self.ptm_to_mark.items():
            self.mark_to_ptm[v] = k

        self.residual_to_mass = {
            "G": 57.021464,
            "A": 71.037114,
            "S": 87.032028,
            "P": 97.052764,
            "V": 99.068414,
            "T": 101.047670,
            "0": 160.030649,  # 103.009185 + 57.021464
            "L": 113.084064,
            "I": 113.084064,
            "N": 114.042927,
            "D": 115.026943,
            "Q": 128.058578,
            "K": 128.094963,
            "E": 129.042593,
            "M": 131.040485,
            "H": 137.058912,
            "F": 147.068414,
            "R": 156.101111,
            "Y": 163.063329,
            "W": 186.079313,
            # Amino acid modifications.
            "1": 147.035400,    # Met oxidation:   131.040485 + 15.994915
            "2": 115.026943,     # Asn deamidation: 114.042927 +  0.984016
            "3": 129.042594,     # Gln deamidation: 128.058578 +  0.984016
            # N-terminal modifications.
            "4": 42.010565,      # Acetylation
            "6": 43.005814,      # Carbamylation
            "7": -17.026549,     # NH3 loss
            "5": 25.980265
        }
        
        self.residual_to_unit_mass = {}
        for k, v in self.residual_to_mass.items():
            self.residual_to_unit_mass[k] = v / self.gap_penalty
        
        mass_list = []
        for key, value in self.residual_to_unit_mass.items():
            if key != "I": # only need one between L and I, otherwise the search space is limited
                mass_list.append((key, value))
        mass_list.sort(key= lambda x: x[1])

        self.residual_order = [x[0] for x in mass_list]
        self.mass_order = [x[1] for x in mass_list]

        mid_mass_list = []
        for key, value in self.residual_to_unit_mass.items():
            if key not in ["4", "5", "6", "7"]:
                mid_mass_list.append((key, value))
        mid_mass_list.sort(key= lambda x: x[1])

        self.mid_residual_order = [x[0] for x in mid_mass_list]
        self.mid_mass_order = [x[1] for x in mid_mass_list]


        self.residual_to_idx = {}
        for i, r in enumerate(self.residual_order):
            self.residual_to_idx[r] = i
        self.diff_mass = np.zeros((len(self.residual_order), len(self.residual_order)))
        for i in range(len(self.residual_order)):
            for j in range(len(self.residual_order)):
                self.diff_mass[i, j] = self.mass_order[i] - self.mass_order[j]
    
    def replace_ptm(self, sequences):
        if type(sequences) == list:
            replaced_sequences = []
            for seq in sequences:
                for ptm, mark in self.ptm_to_mark.items():
                    seq = seq.replace(ptm, mark)
                replaced_sequences.append(seq)
            return replaced_sequences
        elif type(sequences) == str:
            for ptm, mark in self.ptm_to_mark.items():
                sequences = sequences.replace(ptm, mark)
            return sequences

    def replace_mark(self, sequences):
        replaced_sequences = []
        for seq in sequences:
            seq = list(seq)
            seq = [self.mark_to_ptm[s] if s in self.mark_to_ptm.keys() else s for s in seq]
            seq = "".join(seq)
            replaced_sequences.append(seq)
        return replaced_sequences
    
    def TMmass(self, query, target, trace=False):
        L_query = len(query)
        L_target = len(target)

        query_mass_list = []
        for i in range(L_query):
            query_mass_list.append(self.residual_to_unit_mass[query[i]])
        target_mass_list = []
        for i in range(L_target):
            target_mass_list.append(self.residual_to_unit_mass[target[i]])

        dp = np.zeros((L_query + 1, L_target + 1))
        path = np.zeros((L_query + 1, L_target + 1), dtype=int)

        for i in range(1, L_query + 1):
            dp[i, 0] = i
        for j in range(1, L_target + 1):
            dp[0, j] = j
        

        for i in range(1, L_query + 1):
            for j in range(1, L_target + 1):
                q_mass = query_mass_list[i-1]
                t_mass = target_mass_list[j-1]
                diff = abs(q_mass - t_mass)
               
                choices = [
                    (dp[i - 1][j] + 1, 1), 
                    (dp[i][j - 1] + 1, 2),  
                    (dp[i - 1][j - 1] + diff, 3)     
                ]
                
                dp[i][j], path[i][j] = min(choices)
        if not trace:
            return dp[L_query][L_target], None
        else:
            i, j = L_query, L_target
            matching_path = []
            diff_list = []
            while i > 0 or j > 0:
                if path[i][j] == 3:
                    matching_path.append((query[i - 1], target[j - 1]))
                    diff_list.append(dp[i][j])
                    i -= 1
                    j -= 1
                elif path[i][j] == 1:
                    matching_path.append((query[i - 1], '-'))
                    diff_list.append(dp[i][j])
                    i -= 1
                elif path[i][j] == 2:
                    matching_path.append(('-', target[j - 1])) 
                    j -= 1
            diff_list.reverse()
            matching_path.reverse()
            return dp[L_query][L_target], np.array(diff_list)
        
    def Greedymass(self, query, target):
        L_query = len(query)
        L_target = len(target)

        query_mass_list = []
        for i in range(L_query):
            query_mass_list.append(self.residual_to_unit_mass[query[i]])
        target_mass_list = []
        for i in range(L_target):
            target_mass_list.append(self.residual_to_unit_mass[target[i]])
        
        query_mass = np.cumsum(np.array(query_mass_list))
        target_mass = np.cumsum(np.array(target_mass_list))

        query_bias = []
        for i in range(len(query_mass)):
            q = query_mass[i]
            t_ = np.argmin(np.abs(q - target_mass.copy()))
            query_bias.append(q - target_mass[t_])
        return np.array(query_bias)
    

    def MSAFormater(self, sequences):
        # sequences: list of list of string
        sequences = [self.replace_ptm(s) for s in sequences]
        def pad_to_longest(query, target):
            num = len(target) - len(query)
            return query + '-' * num
        # calculate label
        msa_list = []
        for sequence_list in sequences:
            target = sequence_list[0]
            max_index = max(range(len(sequence_list)), key=lambda i: len(sequence_list[i]))
            longest_seq = sequence_list[max_index]
            msa = [((self.TMmass(seq, target, trace=False)[0], self.Greedymass(seq, target)), pad_to_longest(seq, longest_seq)) for i, seq in enumerate(sequence_list)]
            msa_list.append(msa)
        return msa_list



agent = MassAgent()
    
class Alphabet(object):
    def __init__(
        self,
        standard_toks: Sequence[str],
        prepend_toks: Sequence[str] = ("<cls>", "<pad>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<mask>",),
        prepend_bos: bool = True,
        append_eos: bool = False,
        use_msa: bool = True,
    ):
        standard_toks = proteinseq_toks["toks"]
        prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
        append_toks = ("<mask>",)
        prepend_bos = True
        append_eos = False
        use_msa = True
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.use_msa = use_msa

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        for i in range((8 - (len(self.all_toks) % 8)) % 8):
            self.all_toks.append(f"<null_{i  + 1}>")
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}
        self.token_to_mass = {}
        for tok in self.all_toks:
            self.token_to_mass[tok] = agent.residual_to_mass[tok] if tok in agent.residual_to_mass.keys() else 0
        
        self.idx_to_mass = {self.tok_to_idx[tok]: self.token_to_mass[tok] for tok in self.all_toks}
        
        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")
        self.all_special_tokens = ['<eos>', '<unk>', '<pad>', '<cls>', '<mask>']
        self.unique_no_split_tokens = self.all_toks

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def get_batch_converter(self, truncation_seq_length: int = None):
        if self.use_msa:
            return MSABatchConverter(self, truncation_seq_length)
        else:
            return BatchConverter(self, truncation_seq_length)

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:
        """
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
              
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in self.tokenize(text)]


class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet, truncation_seq_length: int = None):
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, raw_batch: Sequence[Tuple[float, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        raw_batch_, seq_str_list = zip(*raw_batch)
        batch_labels, batch_fine_labels = zip(*raw_batch_)

        batch_fine_labels = [torch.from_numpy(s) for s in batch_fine_labels]
        batch_fine_mask = [torch.ones_like(s, dtype=torch.bool) for s in batch_fine_labels]
        batch_fine_labels = rnn_utils.pad_sequence(batch_fine_labels, batch_first=True, padding_value=0)
        batch_fine_mask = rnn_utils.pad_sequence(batch_fine_mask, batch_first=True, padding_value=False)

        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list]
        if self.truncation_seq_length:
            seq_encoded_list = [seq_str[:self.truncation_seq_length] for seq_str in seq_encoded_list]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        tokens = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, (label, seq_str, seq_encoded) in enumerate(
            zip(batch_labels, seq_str_list, seq_encoded_list)
        ):
            labels.append(label)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[
                i,
                int(self.alphabet.prepend_bos) : len(seq_encoded)
                + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx

        return labels, strs, tokens, batch_fine_labels, batch_fine_mask
    

    


class MSABatchConverter(BatchConverter):
    def __call__(self, inputs: Union[Sequence[RawMSA], RawMSA]):
        if isinstance(inputs[0][0], str):
            # Input is a single MSA
            raw_batch: Sequence[RawMSA] = [inputs]  # type: ignore
        else:
            raw_batch = inputs  # type: ignore

        batch_size = len(raw_batch)
        max_alignments = max(len(msa) for msa in raw_batch)
        max_seqlen = max(len(msa[0][1]) for msa in raw_batch)

        tokens = torch.empty(
            (
                batch_size,
                max_alignments,
                max_seqlen + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        fine_labels = torch.zeros((batch_size, max_alignments, 
                                   max_seqlen + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos)))
        fine_mask = torch.zeros_like(tokens, dtype=torch.bool)
        labels = []
        strs = []

        for i, msa in enumerate(raw_batch):
            
            msa_seqlens = set(len(seq) for _, seq in msa)
            if not len(msa_seqlens) == 1:
                raise RuntimeError(
                    "Received unaligned sequences for input to MSA, all sequence "
                    "lengths must be equal."
                )
            msa_labels, msa_strs, msa_tokens, msa_fine_labels, msa_fine_mask = super().__call__(msa)
            msa_labels = msa_labels + [-1] * (max_alignments - len(msa_labels))
            labels.append(msa_labels)
            strs.append(msa_strs)
            tokens[i, : msa_tokens.size(0), : msa_tokens.size(1)] = msa_tokens
            fine_labels[i, : msa_fine_labels.size(0), int(self.alphabet.prepend_bos): int(self.alphabet.prepend_bos)+ msa_fine_labels.size(1)] = msa_fine_labels
            fine_mask[i, : msa_fine_mask.size(0), int(self.alphabet.prepend_bos): int(self.alphabet.prepend_bos)+ msa_fine_mask.size(1)] = msa_fine_mask
        # mask = (tokens == self.alphabet.padding_idx).all(dim=2)
        labels = torch.tensor(labels)
        return labels, strs, tokens, fine_labels, fine_mask
