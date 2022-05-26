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
        letter_case (str): There are three options to alter the letter cases
            of gt texts:

            - unchanged: Do not change gt texts.
            - upper: Convert gt texts into uppercase characters.
            - lower: Convert gt texts into lowercase characters.
            Usually, it only works for English characters. Defaults to
            'unchanged'.
    """

    def __init__(self,
                 dictionary: Union[Dict, Dictionary],
                 max_seq_len: int = 40,
                 letter_case: str = 'unchanged',
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
        assert letter_case in ['unchanged', 'upper', 'lower']
        self.letter_case = letter_case

    def get_targets(
        self, data_samples: Sequence[TextRecogDataSample]
    ) -> Sequence[TextRecogDataSample]:
        """Target generator.

        Args:
            data_samples (list[TextRecogDataSample]): It usually includes
                ``gt_text`` information.

        Returns:
            list[TextRecogDataSample]: Updated data_samples. Two keys will be
            added to data_sample:

            - indexes (torch.LongTensor): Character indexes representing gt
              texts.
            - padding_indexes (torch.LongTensor) Character indexes
              representing gt texts, following several padding_idxs until
              reaching the length of ``max_seq_len``.
        """

        for data_sample in data_samples:
            text = data_sample.gt_text.item
            if self.letter_case in ['upper', 'lower']:
                text = getattr(text, self.letter_case)()
            indexes = self.dictionary.str2idx(text)
            indexes = torch.LongTensor(indexes)

            # target indexes for loss
            src_target = torch.LongTensor(indexes.size(0) + 2).fill_(0)
            src_target[1:-1] = indexes
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
                padding_indexes = (torch.ones(self.max_seq_len) *
                                   self.dictionary.padding_idx).long()
                char_num = min(src_target.size(0), self.max_seq_len)
                padding_indexes[:char_num] = src_target[:char_num]
            else:
                padding_indexes = src_target

            # put in DataSample
            data_sample.gt_text.indexes = indexes
            data_sample.gt_text.padding_indexes = padding_indexes
        return data_samples
