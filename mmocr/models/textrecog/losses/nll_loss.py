# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmocr.models.builder import LOSSES


@LOSSES.register_module()
class NLLLoss(nn.Module):
    """Implementation of loss module for encoder-decoder based text recognition
    method with negative log likelihood loss.

    Args:
        ignore_index (int): Specifies a target value that is
            ignored and does not contribute to the input gradient.
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ('none', 'mean', 'sum').
        ignore_first_char (bool): Whether to ignore the first token in target (
            usually the start token). If ``True``, the last token of the output
            sequence will also be removed to be aligned with the target length.
    """

    def __init__(self,
                 ignore_index=-1,
                 reduction='none',
                 ignore_first_char=False):
        super().__init__()
        assert isinstance(ignore_index, int)
        assert isinstance(reduction, str)
        assert reduction in ['none', 'mean', 'sum']
        assert isinstance(ignore_first_char, bool)

        self.loss_nll = nn.NLLLoss(
            ignore_index=ignore_index, reduction=reduction)
        self.ignore_first_char = ignore_first_char

    def format(self, outputs, targets_dict):
        targets = targets_dict['padded_targets']
        if self.ignore_first_char:
            targets = targets[:, 1:].contiguous()
            outputs = outputs[:, :-1, :]

        outputs = outputs.permute(0, 2, 1).contiguous()

        return outputs, targets

    def forward(self, outputs, targets_dict, img_metas=None):
        """
        Args:
            outputs (Tensor): A raw logit tensor of shape :math:`(N, T, C)`.
            targets_dict (dict): A dict with a key ``padded_targets``, which is
                a tensor of shape :math:`(N, T)`. Each element is the index of
                a character.
            img_metas (None): Unused.

        Returns:
            dict: A loss dict with the key ``loss_ce``.
        """
        outputs, targets = self.format(outputs, targets_dict)

        loss_nll = self.loss_nll(outputs, targets.to(outputs.device))
        losses = dict(loss_nll=loss_nll)

        return losses
