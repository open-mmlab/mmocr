# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Sequence, Union

import torch
import torch.nn as nn

from mmocr.data import TextRecogDataSample
from mmocr.models.textrecog.dictionary.dictionary import Dictionary
from mmocr.registry import MODELS
from .base_recog_module_loss import BaseRecogModuleLoss


@MODELS.register_module()
class CEModuleLoss(BaseRecogModuleLoss):
    """Implementation of loss module for encoder-decoder based text recognition
    method with CrossEntropy loss.

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
        ignore_char (int or str): Specifies a target value that is
            ignored and does not contribute to the input gradient.
            ignore_char can be int or str. If int, it is the index of
            the ignored char. If str, it is the character to ignore.
            Apart from single characters, each item can be one of the
            following reversed keywords: 'padding', 'start', 'end',
            and 'unknown', which refer to their corresponding special
            tokens in the dictionary. It will not ignore any special
            tokens when ignore_char == -1 or 'none'. Defaults to 'padding'.
        flatten (bool): Whether to flatten the output and target before
            computing CE loss. Defaults to False.
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ('none', 'mean', 'sum'). Defaults
            to 'none'.
        ignore_first_char (bool): Whether to ignore the first token in target (
            usually the start token). If ``True``, the last token of the output
            sequence will also be removed to be aligned with the target length.
            Defaults to ``False``.
        flatten (bool): Whether to flatten the vectors for loss computation.
            Defaults to False.
    """

    def __init__(self,
                 dictionary: Union[Dict, Dictionary],
                 max_seq_len: int = 40,
                 letter_case: str = 'unchanged',
                 ignore_char: Union[int, str] = 'padding',
                 flatten: bool = False,
                 reduction: str = 'none',
                 ignore_first_char: bool = False):
        super().__init__(
            dictionary=dictionary,
            max_seq_len=max_seq_len,
            letter_case=letter_case)
        assert isinstance(ignore_char, (int, str))
        assert isinstance(reduction, str)
        assert reduction in ['none', 'mean', 'sum']
        assert isinstance(ignore_first_char, bool)
        assert isinstance(flatten, bool)
        self.flatten = flatten

        self.ignore_first_char = ignore_first_char

        if isinstance(ignore_char, int):
            ignore_index = ignore_char
        else:
            mapping_table = {
                'none': -1,
                'start': self.dictionary.start_idx,
                'padding': self.dictionary.padding_idx,
                'end': self.dictionary.end_idx,
                'unknown': self.dictionary.unknown_idx,
            }
            # TODO add char2id in Dictionary
            ignore_index = mapping_table.get(
                ignore_char, self.dictionary._char2idx.get(ignore_char, None))
            if ignore_index is None:
                warnings.warn(
                    f'{ignore_char} does not exist in the dictionary',
                    UserWarning)
                ignore_index = -1

        self.ignore_char = ignore_char
        self.ignore_index = ignore_index
        self.loss_ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction=reduction)

    def forward(self, outputs: torch.Tensor,
                data_samples: Sequence[TextRecogDataSample]) -> Dict:
        """
        Args:
            outputs (Tensor): A raw logit tensor of shape :math:`(N, T, C)`.
            data_samples (list[TextRecogDataSample]): List of
                ``TextRecogDataSample`` which are processed by ``get_target``.

        Returns:
            dict: A loss dict with the key ``loss_ce``.
        """
        targets = list()
        for data_sample in data_samples:
            targets.append(data_sample.gt_text.padded_indexes)
        targets = torch.stack(targets, dim=0).long()
        if self.ignore_first_char:
            targets = targets[:, 1:].contiguous()
            outputs = outputs[:, :-1, :].contiguous()
        if self.flatten:
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
        else:
            outputs = outputs.permute(0, 2, 1).contiguous()

        loss_ce = self.loss_ce(outputs, targets.to(outputs.device))
        losses = dict(loss_ce=loss_ce)

        return losses
