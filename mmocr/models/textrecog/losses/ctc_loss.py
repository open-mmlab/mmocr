import torch
import torch.nn as nn

from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class CTCLoss(nn.Module):
    """Implementation of loss module for CTC-loss based text recognition.

    Args:
        flatten (bool): If True, use flattened targets, else padded targets.
        blank (int): Blank label. Default 0.
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ('none', 'mean', 'sum').
        zero_infinity (bool): Whether to zero infinite losses and
            the associated gradients. Default: False.
            Infinite losses mainly occur when the inputs
            are too short to be aligned to the targets.
    """

    def __init__(self,
                 flatten=True,
                 blank=0,
                 reduction='mean',
                 zero_infinity=False,
                 **kwargs):
        super().__init__()
        assert isinstance(flatten, bool)
        assert isinstance(blank, int)
        assert isinstance(reduction, str)
        assert isinstance(zero_infinity, bool)

        self.flatten = flatten
        self.blank = blank
        self.ctc_loss = nn.CTCLoss(
            blank=blank, reduction=reduction, zero_infinity=zero_infinity)

    def forward(self, outputs, targets_dict):

        outputs = torch.log_softmax(outputs, dim=2)
        bsz, seq_len = outputs.size(0), outputs.size(1)
        input_lengths = torch.full(
            size=(bsz, ), fill_value=seq_len, dtype=torch.long)
        outputs_for_loss = outputs.permute(1, 0, 2).contiguous()  # T * N * C

        if self.flatten:
            targets = targets_dict['flatten_targets']
        else:
            targets = torch.full(
                size=(bsz, seq_len), fill_value=self.blank, dtype=torch.long)
            for idx, tensor in enumerate(targets_dict['targets']):
                valid_len = min(tensor.size(0), seq_len)
                targets[idx, :valid_len] = tensor[:valid_len]

        target_lengths = targets_dict['target_lengths']

        loss_ctc = self.ctc_loss(outputs_for_loss, targets, input_lengths,
                                 target_lengths)

        losses = dict(loss_ctc=loss_ctc)

        return losses
