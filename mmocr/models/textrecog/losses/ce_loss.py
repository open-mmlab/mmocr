# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence, Union

import torch
import torch.nn as nn

from mmocr.core import TextRecogDataSample
from mmocr.models.textrecog.dictionary.dictionary import Dictionary
from mmocr.registry import MODELS
from .base_recog_loss import BaseRecogLoss


@MODELS.register_module()
class CELoss(BaseRecogLoss):
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
        ignore_index (int): Specifies a target value that is
            ignored and does not contribute to the input gradient. Defaults to
            -1.
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
                 ignore_index: int = -1,
                 flatten: bool = False,
                 reduction: str = 'none',
                 ignore_first_char: bool = False):
        super().__init__(
            dictionary=dictionary,
            max_seq_len=max_seq_len,
            letter_case=letter_case)
        assert isinstance(ignore_index, int)
        assert isinstance(reduction, str)
        assert reduction in ['none', 'mean', 'sum']
        assert isinstance(ignore_first_char, bool)
        assert isinstance(flatten, bool)
        self.flatten = flatten
        self.loss_ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction=reduction)
        self.ignore_first_char = ignore_first_char

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
