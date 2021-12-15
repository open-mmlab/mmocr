# Copyright (c) OpenMMLab. All rights reserved.
import torch

import mmocr.utils as utils
from mmocr.models.builder import CONVERTORS
from .attn import AttnConvertor


@CONVERTORS.register_module()
class ABIConvertor(AttnConvertor):
    """Convert between text, index and tensor for encoder-decoder based
    pipeline. Modified from AttnConvertor to get closer to ABINet's original
    implementation.

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
    """

    def str2tensor(self, strings):
        """
        Convert text-string into tensor. Different from
        :obj:`mmocr.models.textrecog.convertors.AttnConvertor`, the targets
        field returns target index no longer than max_seq_len (EOS token
        included).

        Args:
            strings (list[str]): For instance, ['hello', 'world']

        Returns:
            dict: A dict with two tensors.

            - | targets (list[Tensor]): [torch.Tensor([1,2,3,3,4,8]),
                torch.Tensor([5,4,6,3,7,8])]
            - | padded_targets (Tensor): Tensor of shape
                (bsz * max_seq_len)).
        """
        assert utils.is_type_list(strings, str)

        tensors, padded_targets = [], []
        indexes = self.str2idx(strings)
        for index in indexes:
            tensor = torch.LongTensor(index[:self.max_seq_len - 1] +
                                      [self.end_idx])
            tensors.append(tensor)
            # target tensor for loss
            src_target = torch.LongTensor(tensor.size(0) + 1).fill_(0)
            src_target[0] = self.start_idx
            src_target[1:] = tensor
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
