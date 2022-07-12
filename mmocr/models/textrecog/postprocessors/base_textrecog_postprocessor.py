# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Optional, Sequence, Tuple, Union

import mmcv
import torch
from mmengine.data import LabelData

from mmocr.data import TextRecogDataSample
from mmocr.models.textrecog.dictionary import Dictionary
from mmocr.registry import TASK_UTILS


class BaseTextRecogPostprocessor:
    """Base text recognition postprocessor.

    Args:
        dictionary (dict or :obj:`Dictionary`): The config for `Dictionary` or
            the instance of `Dictionary`.
        max_seq_len (int): max_seq_len (int): Maximum sequence length. The
            sequence is usually generated from decoder. Defaults to 40.
        ignore_chars (list[str]): A list of characters to be ignored from the
            final results. Postprocessor will skip over these characters when
            converting raw indexes to characters. Apart from single characters,
            each item can be one of the following reversed keywords: 'padding',
            'end' and 'unknown', which refer to their corresponding special
            tokens in the dictionary.
    """

    def __init__(self,
                 dictionary: Union[Dictionary, Dict],
                 max_seq_len: int = 40,
                 ignore_chars: Sequence[str] = ['padding'],
                 **kwargs) -> None:

        if isinstance(dictionary, dict):
            self.dictionary = TASK_UTILS.build(dictionary)
        elif isinstance(dictionary, Dictionary):
            self.dictionary = dictionary
        else:
            raise TypeError(
                'The type of dictionary should be `Dictionary` or dict, '
                f'but got {type(dictionary)}')
        self.max_seq_len = max_seq_len

        mapping_table = {
            'padding': self.dictionary.padding_idx,
            'end': self.dictionary.end_idx,
            'unknown': self.dictionary.unknown_idx,
        }
        if not mmcv.is_list_of(ignore_chars, str):
            raise TypeError('ignore_chars must be list of str')
        ignore_indexes = list()
        for ignore_char in ignore_chars:
            # TODO add char2id in Dictionary
            index = mapping_table.get(
                ignore_char, self.dictionary._char2idx.get(ignore_char, None))
            if index is None:
                warnings.warn(
                    f'{ignore_char} does not exist in the dictionary',
                    UserWarning)
                continue
            ignore_indexes.append(index)
        self.ignore_indexes = ignore_indexes

    def get_single_prediction(
        self,
        output: torch.Tensor,
        data_sample: Optional[TextRecogDataSample] = None,
    ) -> Tuple[Sequence[int], Sequence[float]]:
        """Convert the output of a single image to index and score.

        Args:
           output (torch.Tensor): Single image output.
           data_sample (TextRecogDataSample): Datasample of an image.

        Returns:
            tuple(list[int], list[float]): index and score.
        """
        raise NotImplementedError

    def __call__(
        self, outputs: torch.Tensor,
        data_samples: Sequence[TextRecogDataSample]
    ) -> Sequence[TextRecogDataSample]:
        """Convert outputs to strings and scores.

        Args:
            outputs (torch.Tensor): The model outputs in size: N * T * C
            data_samples (list[TextRecogDataSample]): The list of
                TextRecogDataSample.

        Returns:
            list(TextRecogDataSample): The list of TextRecogDataSample. It
            usually contain ``pred_text`` information.
        """
        batch_size = outputs.size(0)

        for idx in range(batch_size):
            index, score = self.get_single_prediction(outputs[idx, :, :],
                                                      data_samples[idx])
            text = self.dictionary.idx2str(index)
            pred_text = LabelData()
            pred_text.score = score
            pred_text.item = text
            data_samples[idx].pred_text = pred_text
        return data_samples
