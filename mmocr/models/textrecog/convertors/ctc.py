import math

import torch
import torch.nn.functional as F

import mmocr.utils as utils
from mmocr.models.builder import CONVERTORS
from .base import BaseConvertor


@CONVERTORS.register_module()
class CTCConvertor(BaseConvertor):
    """Convert between text, index and tensor for CTC loss-based pipeline.

    Args:
        dict_type (str): Type of dict, should be either 'DICT36' or 'DICT90'.
        dict_file (None|str): Character dict file path. If not none, the file
            is of higher priority than dict_type.
        dict_list (None|list[str]): Character list. If not none, the list
            is of higher priority than dict_type, but lower than dict_file.
        with_unknown (bool): If True, add `UKN` token to class.
        lower (bool): If True, convert original string to lower case.
    """

    def __init__(self,
                 dict_type='DICT90',
                 dict_file=None,
                 dict_list=None,
                 with_unknown=True,
                 lower=False,
                 **kwargs):
        super().__init__(dict_type, dict_file, dict_list)
        assert isinstance(with_unknown, bool)
        assert isinstance(lower, bool)

        self.with_unknown = with_unknown
        self.lower = lower
        self.update_dict()

    def update_dict(self):
        # CTC-blank
        blank_token = '<BLK>'
        self.blank_idx = 0
        self.idx2char.insert(0, blank_token)

        # unknown
        self.unknown_idx = None
        if self.with_unknown:
            self.idx2char.append('<UKN>')
            self.unknown_idx = len(self.idx2char) - 1

        # update char2idx
        self.char2idx = {}
        for idx, char in enumerate(self.idx2char):
            self.char2idx[char] = idx

    def str2tensor(self, strings):
        """Convert text-string to ctc-loss input tensor.

        Args:
            strings (list[str]): ['hello', 'world'].
        Returns:
            dict (str: tensor | list[tensor]):
                tensors (list[tensor]): [torch.Tensor([1,2,3,3,4]),
                    torch.Tensor([5,4,6,3,7])].
                flatten_targets (tensor): torch.Tensor([1,2,3,3,4,5,4,6,3,7]).
                target_lengths (tensor): torch.IntTensot([5,5]).
        """
        assert utils.is_type_list(strings, str)

        tensors = []
        indexes = self.str2idx(strings)
        for index in indexes:
            tensor = torch.IntTensor(index)
            tensors.append(tensor)
        target_lengths = torch.IntTensor([len(t) for t in tensors])
        flatten_target = torch.cat(tensors)

        return {
            'targets': tensors,
            'flatten_targets': flatten_target,
            'target_lengths': target_lengths
        }

    def tensor2idx(self, output, img_metas, topk=1, return_topk=False):
        """Convert model output tensor to index-list.
        Args:
            output (tensor): The model outputs with size: N * T * C.
            img_metas (list[dict]): Each dict contains one image info.
            topk (int): The highest k classes to be returned.
            return_topk (bool): Whether to return topk or just top1.
        Returns:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]].
            scores (list[list[float]]): [[0.9,0.8,0.95,0.97,0.94],
                [0.9,0.9,0.98,0.97,0.96]]
                (
                    indexes_topk (list[list[list[int]->len=topk]]):
                    scores_topk (list[list[list[float]->len=topk]])
                ).
        """
        assert utils.is_type_list(img_metas, dict)
        assert len(img_metas) == output.size(0)
        assert isinstance(topk, int)
        assert topk >= 1

        valid_ratios = [
            img_meta.get('valid_ratio', 1.0) for img_meta in img_metas
        ]

        batch_size = output.size(0)
        output = F.softmax(output, dim=2)
        output = output.cpu().detach()
        batch_topk_value, batch_topk_idx = output.topk(topk, dim=2)
        batch_max_idx = batch_topk_idx[:, :, 0]
        scores_topk, indexes_topk = [], []
        scores, indexes = [], []
        feat_len = output.size(1)
        for b in range(batch_size):
            valid_ratio = valid_ratios[b]
            decode_len = min(feat_len, math.ceil(feat_len * valid_ratio))
            pred = batch_max_idx[b, :]
            select_idx = []
            prev_idx = self.blank_idx
            for t in range(decode_len):
                tmp_value = pred[t].item()
                if tmp_value not in (prev_idx, self.blank_idx):
                    select_idx.append(t)
                prev_idx = tmp_value
            select_idx = torch.LongTensor(select_idx)
            topk_value = torch.index_select(batch_topk_value[b, :, :], 0,
                                            select_idx)  # valid_seqlen * topk
            topk_idx = torch.index_select(batch_topk_idx[b, :, :], 0,
                                          select_idx)
            topk_idx_list, topk_value_list = topk_idx.numpy().tolist(
            ), topk_value.numpy().tolist()
            indexes_topk.append(topk_idx_list)
            scores_topk.append(topk_value_list)
            indexes.append([x[0] for x in topk_idx_list])
            scores.append([x[0] for x in topk_value_list])

        if return_topk:
            return indexes_topk, scores_topk

        return indexes, scores
