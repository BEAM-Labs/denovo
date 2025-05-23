# Copyright (c) Puyuan Liu
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List, Tuple
from torch import TensorType
#from fairseq.data.ctc_dictionary import CTCDictionary
import torch


class CTCDecoderBase:
    """
    The base class of CTC decoders
    """
    def __init__(self, decoder) :
        self.attn_decoder = decoder 
        #self.desired_length = decoder_parameters["desired_length"]
        #self.truncate = decoder_parameters["truncate_summary"]

    @staticmethod
    def unravel_indices(indices: torch.LongTensor, shape: Tuple[int, ...]) -> torch.LongTensor:
        """
        Unravel the index of a flatted tensor given its original dimension. This function is copied from torch.test
        """
        coord = []
        for dim in reversed(shape):
            coord.append(indices % dim)
            indices = torch.div(indices, dim, rounding_mode='floor')

        coord = torch.stack(coord[::-1], dim=-1)

        return coord

    def decode(self, log_prob: TensorType, source_length: TensorType) -> List[List[int]]:
        """
        Decode the log of the output probability into (vocab) index.
        This function is expected to return 2-D list, where each sublist gives the index of a decoded sentence.
        """
        raise (NotImplementedError, "The decode function must be overwritten by the subclass")

    def ctc_post_processing(self, sentence_index: List[int]) -> List[int]:
        """
        Merge repetitive tokens, then eliminate <blank> tokens and <pad> tokens.
        The input sentence_index is expected to be a 1-D index list
        """
        sentence_index = self.remove_repentance(sentence_index)
        sentence_index = list(filter((self.attn_decoder.get_blank_idx()).__ne__, sentence_index))
        #sentence_index = list(filter((self.attn_decoder.get_pad_idx()).__ne__, sentence_index))

        return sentence_index

    @staticmethod
    def remove_repentance(index_list: List[int]) -> List[int]:
        """
        Eliminate repeated index in list. e.g., [1, 1, 2, 2, 3] --> [1, 2, 3]
        """
        return [a for a, b in zip(index_list, index_list[1:] + [not index_list[-1]]) if a != b]

    @staticmethod
    def get_topk_cpu():
        """
        Move tensor to CPU, take topk and then move back
        """
