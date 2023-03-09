# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch
import torch.nn as nn

from mmocr import digit_version
from mmocr.registry import MODELS


@MODELS.register_module()
class SmoothL1Loss(nn.SmoothL1Loss):
    """Smooth L1 loss."""


@MODELS.register_module()
class MaskedSmoothL1Loss(nn.Module):
    """Masked Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.
        eps (float, optional): Eps to avoid zero-division error.  Defaults to
            1e-6.
    """

    def __init__(self, beta: Union[float, int] = 1, eps: float = 1e-6) -> None:
        super().__init__()
        if digit_version(torch.__version__) > digit_version('1.6.0'):
            if digit_version(torch.__version__) >= digit_version(
                    '1.13.0') and beta == 0:
                beta = beta + eps
            self.smooth_l1_loss = nn.SmoothL1Loss(beta=beta, reduction='none')
        self.eps = eps
        self.beta = beta

    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction in any shape.
            gt (torch.Tensor): The learning target of the prediction in the
                same shape as pred.
            mask (torch.Tensor, optional): Binary mask in the same shape of
                pred, indicating positive regions to calculate the loss. Whole
                region will be taken into account if not provided. Defaults to
                None.

        Returns:
            torch.Tensor: The loss value.
        """

        assert pred.size() == gt.size() and gt.numel() > 0
        if mask is None:
            mask = torch.ones_like(gt).bool()
        assert mask.size() == gt.size()
        x = pred * mask
        y = gt * mask
        if digit_version(torch.__version__) > digit_version('1.6.0'):
            loss = self.smooth_l1_loss(x, y)
        else:
            loss = torch.zeros_like(gt)
            diff = torch.abs(x - y)
            mask_beta = diff < self.beta
            loss[mask_beta] = 0.5 * torch.square(diff)[mask_beta] / self.beta
            loss[~mask_beta] = diff[~mask_beta] - 0.5 * self.beta
        return loss.sum() / (mask.sum() + self.eps)
