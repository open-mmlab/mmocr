# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, Sequence, Union

import torch
import torch.nn as nn

from mmocr.core.data_structures import TextRecogDataSample
from mmocr.models.textrecog.dictionary import Dictionary
from mmocr.registry import TASK_UTILS


class BaseRecogLoss(nn.Module):
    """Base recognition loss.

    Args:
        dictionary (dict or :obj:`Dictionary`): The config for `Dictionary` or
            the instance of `Dictionary`.
        max_seq_len (int): Maximum sequence length. The sequence is usually
            generated from decoder. Defaults to 40.
        weight (int or float): The weight of loss. Defaults to 1.
    """

    def __init__(self,
                 dictionary: Union[Dict, Dictionary],
                 max_seq_len: int = 40,
                 **kwargs) -> None:
        super().__init__()
        if isinstance(dictionary, dict):
            self.dictionary = TASK_UTILS.build(dictionary)
        elif isinstance(dictionary, Dictionary):
            self.dictionary = dictionary
        else:
            raise TypeError(
                'The type of dictionary should be `Dictionary` or dict, '
                f'but got {type(dictionary)}')
        self.max_seq_len = max_seq_len

    def get_targets(
        self, data_samples: Sequence[TextRecogDataSample]
    ) -> Sequence[TextRecogDataSample]:
        """Target generator.

        Args:
            data_samples (list[TextRecogDataSample]): It usually includes
                ``gt_text`` information.

        Returns:
            list[TextRecogDataSample]: updated data_samples.
        """

        for data_sample in data_samples:
            index = self.dictionary.str2idx(data_sample.gt_text.item)
            tensor = torch.LongTensor(index)

            # target tensor for loss
            src_target = torch.LongTensor(tensor.size(0) + 2).fill_(0)
            src_target[1:-1] = tensor
            if self.dictionary.start_idx is not None:
                src_target[0] = self.dictionary.start_idx
                slice_start = 0
            else:
                slice_start = 1
            if self.dictionary.end_idx is not None:
                src_target[-1] = self.dictionary.end_idx
                slice_end = src_target.size(0)
            else:
                slice_end = src_target.size(0) - 1
            src_target = src_target[slice_start:slice_end]
            if self.dictionary.padding_idx is not None:
                padded_target = (torch.ones(self.max_seq_len) *
                                 self.dictionary.padding_idx).long()
                char_num = min(src_target.size(0), self.max_seq_len)
                padded_target[:char_num] = src_target[:char_num]
            else:
                padded_target = src_target

            # put in DataSample
            data_sample.gt_text.item = tensor
            data_sample.gt_text.item_padded = padded_target
        return data_samples
