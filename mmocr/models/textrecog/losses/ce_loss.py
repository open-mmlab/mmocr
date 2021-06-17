import torch.nn as nn
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class CELoss(nn.Module):
    """Implementation of loss module for encoder-decoder based text recognition
    method with CrossEntropy loss.

    Args:
        ignore_index (int): Specifies a target value that is
            ignored and does not contribute to the input gradient.
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ('none', 'mean', 'sum').
    """

    def __init__(self, ignore_index=-1, reduction='none'):
        super().__init__()
        assert isinstance(ignore_index, int)
        assert isinstance(reduction, str)
        assert reduction in ['none', 'mean', 'sum']

        self.loss_ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction=reduction)

    def format(self, outputs, targets_dict):
        targets = targets_dict['padded_targets']

        return outputs.permute(0, 2, 1).contiguous(), targets

    def forward(self, outputs, targets_dict, img_metas=None):
        outputs, targets = self.format(outputs, targets_dict)

        loss_ce = self.loss_ce(outputs, targets.to(outputs.device))
        losses = dict(loss_ce=loss_ce)

        return losses


@LOSSES.register_module()
class SARLoss(CELoss):
    """Implementation of loss module in `SAR.

    <https://arxiv.org/abs/1811.00751>`_.

    Args:
        ignore_index (int): Specifies a target value that is
            ignored and does not contribute to the input gradient.
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ('none', 'mean', 'sum').
    """

    def __init__(self, ignore_index=0, reduction='mean', **kwargs):
        super().__init__(ignore_index, reduction)

    def format(self, outputs, targets_dict):
        targets = targets_dict['padded_targets']
        # targets[0, :], [start_idx, idx1, idx2, ..., end_idx, pad_idx...]
        # outputs[0, :, 0], [idx1, idx2, ..., end_idx, ...]

        # ignore first index of target in loss calculation
        targets = targets[:, 1:].contiguous()
        # ignore last index of outputs to be in same seq_len with targets
        outputs = outputs[:, :-1, :].permute(0, 2, 1).contiguous()

        return outputs, targets


@LOSSES.register_module()
class TFLoss(CELoss):
    """Implementation of loss module for transformer."""

    def __init__(self,
                 ignore_index=-1,
                 reduction='none',
                 flatten=True,
                 **kwargs):
        super().__init__(ignore_index, reduction)
        assert isinstance(flatten, bool)

        self.flatten = flatten

    def format(self, outputs, targets_dict):
        outputs = outputs[:, :-1, :].contiguous()
        targets = targets_dict['padded_targets']
        targets = targets[:, 1:].contiguous()
        if self.flatten:
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
        else:
            outputs = outputs.permute(0, 2, 1).contiguous()

        return outputs, targets
