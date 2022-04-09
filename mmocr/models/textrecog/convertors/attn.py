# Copyright (c) OpenMMLab. All rights reserved.
import torch

import mmocr.utils as utils
from mmocr.models.builder import CONVERTORS
from .base import BaseConvertor


@CONVERTORS.register_module()
class AttnConvertor(BaseConvertor):
    """Convert between text, index and tensor for encoder-decoder based
    pipeline.

    Args:
        dict_type (str): Type of dict, should be one of {'DICT36', 'DICT90'}.
        dict_file (None|str): Character dict file path. If not none,
            higher priority than dict_type.
        dict_list (None|list[str]): Character list. If not none, higher
            priority than dict_type, but lower than dict_file.
        with_unknown (bool): If True, add `UKN` token to class.
        max_seq_len (int): Maximum sequence length of label.
        lower (bool): If True, convert original string to lower case.
        start_end_same (bool): Whether use the same index for
            start and end token or not. Default: True.
        start_end_same_char (bool): Whether use the same char for
            start and end token or not. Default: True.
    """

    def __init__(self,
                 dict_type='DICT90',
                 dict_file=None,
                 dict_list=None,
                 with_unknown=True,
                 max_seq_len=40,
                 lower=False,
                 start_end_same=True,
                 start_end_same_char=True,
                 **kwargs):
        super().__init__(dict_type, dict_file, dict_list)
        assert isinstance(with_unknown, bool)
        assert isinstance(max_seq_len, int)
        assert isinstance(lower, bool)

        self.with_unknown = with_unknown
        self.max_seq_len = max_seq_len
        self.lower = lower
        self.start_end_same = start_end_same
        self.start_end_same_char = start_end_same_char

        self.update_dict()

    def update_dict(self):
        # unknown
        unknown_token = '<UKN>'
        self.unknown_idx = None
        if self.with_unknown:
            self.idx2char.append(unknown_token)
            self.unknown_idx = len(self.idx2char) - 1

        # BOS/EOS
        if self.start_end_same_char:
            start_end_token = '<BOS/EOS>'
            self.idx2char.append(start_end_token)
            self.start_idx = len(self.idx2char) - 1
            if not self.start_end_same:
                self.idx2char.append(start_end_token)
            self.end_idx = len(self.idx2char) - 1
        else:
            assert self.start_end_same is False
            start_token = '<SOS>'
            end_token = '<EOS>'
            self.idx2char.append(start_token)
            self.start_idx = len(self.idx2char) - 1
            self.idx2char.append(end_token)
            self.end_idx = len(self.idx2char) - 1

        # padding
        padding_token = '<PAD>'
        self.idx2char.append(padding_token)
        self.padding_idx = len(self.idx2char) - 1

        # update char2idx
        self.char2idx = {}
        for idx, char in enumerate(self.idx2char):
            self.char2idx[char] = idx

    def str2tensor(self, strings):
        """
        Convert text-string into tensor.
        Args:
            strings (list[str]): ['hello', 'world']
        Returns:
            dict (str: Tensor | list[tensor]):
                tensors (list[Tensor]): [torch.Tensor([1,2,3,3,4]),
                                                    torch.Tensor([5,4,6,3,7])]
                padded_targets (Tensor(bsz * max_seq_len))
        """
        assert utils.is_type_list(strings, str)

        tensors, padded_targets = [], []
        indexes = self.str2idx(strings)
        for index in indexes:
            tensor = torch.LongTensor(index)
            tensors.append(tensor)
            # target tensor for loss
            src_target = torch.LongTensor(tensor.size(0) + 2).fill_(0)
            src_target[-1] = self.end_idx
            src_target[0] = self.start_idx
            src_target[1:-1] = tensor
            padded_target = (torch.ones(self.max_seq_len) *
                             self.padding_idx).long()
            char_num = src_target.size(0)
            if char_num > self.max_seq_len:
                padded_target = src_target[:self.max_seq_len]
            else:
                padded_target[:char_num] = src_target
            padded_targets.append(padded_target)
        padded_targets = torch.stack(padded_targets, 0).long()

        return {'targets': tensors, 'padded_targets': padded_targets}

    def tensor2idx(self, outputs, img_metas=None):
        """
        Convert output tensor to text-index
        Args:
            outputs (tensor): model outputs with size: N * T * C
            img_metas (list[dict]): Each dict contains one image info.
        Returns:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]]
            scores (list[list[float]]): [[0.9,0.8,0.95,0.97,0.94],
                                         [0.9,0.9,0.98,0.97,0.96]]
        """
        batch_size = outputs.size(0)
        ignore_indexes = [self.padding_idx]
        indexes, scores = [], []
        for idx in range(batch_size):
            seq = outputs[idx, :, :]
            max_value, max_idx = torch.max(seq, -1)
            str_index, str_score = [], []
            output_index = max_idx.cpu().detach().numpy().tolist()
            output_score = max_value.cpu().detach().numpy().tolist()
            for char_index, char_score in zip(output_index, output_score):
                if char_index in ignore_indexes:
                    continue
                if char_index == self.end_idx:
                    break
                str_index.append(char_index)
                str_score.append(char_score)

            indexes.append(str_index)
            scores.append(str_score)

        return indexes, scores
