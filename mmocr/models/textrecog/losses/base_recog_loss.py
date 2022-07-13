# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, Sequence, Union

import torch
import torch.nn as nn

from mmocr.data import TextRecogDataSample
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
        pad_with (str): The padding strategy for ``gt_text.padded_indexes``.
            Defaults to 'auto'. Options are:
            - 'auto': Use dictionary.padding_idx to pad gt texts, or
              dictionary.end_idx if dictionary.padding_idx
              is None.
            - 'padding': Always use dictionary.padding_idx to pad gt texts.
            - 'end': Always use dictionary.end_idx to pad gt texts.
            - 'none': Do not pad gt texts.
    """

    def __init__(self,
                 dictionary: Union[Dict, Dictionary],
                 max_seq_len: int = 40,
                 letter_case: str = 'unchanged',
                 pad_with: str = 'auto',
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

        assert pad_with in ['auto', 'padding', 'end', 'none']
        if pad_with == 'auto':
            self.pad_idx = self.dictionary.padding_idx or \
                self.dictionary.end_idx
        elif pad_with == 'padding':
            self.pad_idx = self.dictionary.padding_idx
        elif pad_with == 'end':
            self.pad_idx = self.dictionary.end_idx
        else:
            self.pad_idx = None
        if self.pad_idx is None and pad_with != 'none':
            if pad_with == 'auto':
                raise ValueError('pad_with="auto", but dictionary.end_idx'
                                 ' and dictionary.padding_idx are both None')
            else:
                raise ValueError(
                    f'pad_with="{pad_with}", but dictionary.{pad_with}_idx is'
                    ' None')

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
              texts. All special tokens are excluded, except for UKN.
            - padded_indexes (torch.LongTensor): Character indexes
              representing gt texts with BOS and EOS if applicable, following
              several padding indexes until the length reaches ``max_seq_len``.
              In particular, if ``pad_with='none'``, no padding will be
              applied.
        """

        for data_sample in data_samples:
            if data_sample.get('have_target', False):
                continue
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
            if self.pad_idx is not None:
                padded_indexes = (torch.ones(self.max_seq_len) *
                                  self.pad_idx).long()
                char_num = min(src_target.size(0), self.max_seq_len)
                padded_indexes[:char_num] = src_target[:char_num]
            else:
                padded_indexes = src_target
            # put in DataSample
            data_sample.gt_text.indexes = indexes
            data_sample.gt_text.padded_indexes = padded_indexes
            data_sample.set_metainfo(dict(have_target=True))
        return data_samples
