# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, Sequence, Union

import torch
import torch.nn as nn

from mmocr.models.common.dictionary import Dictionary
from mmocr.registry import MODELS
from mmocr.structures import TextRecogDataSample
from .base import BaseTextRecogModuleLoss


@MODELS.register_module()
class CTCModuleLoss(BaseTextRecogModuleLoss):
    """Implementation of loss module for CTC-loss based text recognition.

    Args:
        dictionary (dict or :obj:`Dictionary`): The config for `Dictionary` or
            the instance of `Dictionary`.
        letter_case (str): There are three options to alter the letter cases
            of gt texts:
            - unchanged: Do not change gt texts.
            - upper: Convert gt texts into uppercase characters.
            - lower: Convert gt texts into lowercase characters.
            Usually, it only works for English characters. Defaults to
            'unchanged'.
        flatten (bool): If True, use flattened targets, else padded targets.
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ('none', 'mean', 'sum').
        zero_infinity (bool): Whether to zero infinite losses and
            the associated gradients. Default: False.
            Infinite losses mainly occur when the inputs
            are too short to be aligned to the targets.
    """

    def __init__(self,
                 dictionary: Union[Dict, Dictionary],
                 letter_case: str = 'unchanged',
                 flatten: bool = True,
                 reduction: str = 'mean',
                 zero_infinity: bool = False,
                 **kwargs) -> None:
        super().__init__(dictionary=dictionary, letter_case=letter_case)
        assert isinstance(flatten, bool)
        assert isinstance(reduction, str)
        assert isinstance(zero_infinity, bool)

        self.flatten = flatten
        self.ctc_loss = nn.CTCLoss(
            blank=self.dictionary.padding_idx,
            reduction=reduction,
            zero_infinity=zero_infinity)

    def forward(self, outputs: torch.Tensor,
                data_samples: Sequence[TextRecogDataSample]) -> Dict:
        """
        Args:
            outputs (Tensor): A raw logit tensor of shape :math:`(N, T, C)`.
            data_samples (list[TextRecogDataSample]): List of
                ``TextRecogDataSample`` which are processed by ``get_target``.

        Returns:
            dict: The loss dict with key ``loss_ctc``.
        """
        valid_ratios = None
        if data_samples is not None:
            valid_ratios = [
                img_meta.get('valid_ratio', 1.0) for img_meta in data_samples
            ]

        outputs = torch.log_softmax(outputs, dim=2)
        bsz, seq_len = outputs.size(0), outputs.size(1)
        outputs_for_loss = outputs.permute(1, 0, 2).contiguous()  # T * N * C
        targets = [
            data_sample.gt_text.indexes[:seq_len]
            for data_sample in data_samples
        ]
        target_lengths = torch.IntTensor([len(t) for t in targets])
        target_lengths = torch.clamp(target_lengths, min=1, max=seq_len).long()
        input_lengths = torch.full(
            size=(bsz, ), fill_value=seq_len, dtype=torch.long)
        if self.flatten:
            targets = torch.cat(targets)
        else:
            padded_targets = torch.full(
                size=(bsz, seq_len),
                fill_value=self.dictionary.padding_idx,
                dtype=torch.long)
            for idx, valid_len in enumerate(target_lengths):
                padded_targets[idx, :valid_len] = targets[idx][:valid_len]
            targets = padded_targets

            if valid_ratios is not None:
                input_lengths = [
                    math.ceil(valid_ratio * seq_len)
                    for valid_ratio in valid_ratios
                ]
                input_lengths = torch.Tensor(input_lengths).long()
        loss_ctc = self.ctc_loss(outputs_for_loss, targets, input_lengths,
                                 target_lengths)
        losses = dict(loss_ctc=loss_ctc)

        return losses

    def get_targets(
        self, data_samples: Sequence[TextRecogDataSample]
    ) -> Sequence[TextRecogDataSample]:
        """Target generator.

        Args:
            data_samples (list[TextRecogDataSample]): It usually includes
                ``gt_text`` information.

        Returns:

            list[TextRecogDataSample]: updated data_samples. It will add two
            key in data_sample:

            - indexes (torch.LongTensor): The index corresponding to the item.
        """

        for data_sample in data_samples:
            text = data_sample.gt_text.item
            if self.letter_case in ['upper', 'lower']:
                text = getattr(text, self.letter_case)()
            indexes = self.dictionary.str2idx(text)
            indexes = torch.IntTensor(indexes)
            data_sample.gt_text.indexes = indexes
        return data_samples
