# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn

from mmocr.registry import MODELS


@MODELS.register_module()
class MaskedDiceLoss(nn.Module):
    """Masked dice loss.

    Args:
        eps (float, optional): Eps to avoid zero-divison error.  Defaults to
            1e-6.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        assert isinstance(eps, float)
        self.eps = eps

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
            mask = torch.ones_like(gt)
        assert mask.size() == gt.size()

        pred = pred.contiguous().view(pred.size(0), -1)
        gt = gt.contiguous().view(gt.size(0), -1)

        mask = mask.contiguous().view(mask.size(0), -1)
        pred = pred * mask
        gt = gt * mask

        dice_coeff = (2 * (pred * gt).sum()) / (
            pred.sum() + gt.sum() + self.eps)

        return 1 - dice_coeff


@MODELS.register_module()
class MaskedSquareDiceLoss(nn.Module):
    """Masked square dice loss.

    Args:
        eps (float, optional): Eps to avoid zero-divison error.  Defaults to
            1e-3.
    """

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        assert isinstance(eps, float)
        self.eps = eps

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
            mask = torch.ones_like(gt)
        assert mask.size() == gt.size()
        batch_size = pred.size(0)
        pred = pred.contiguous().view(batch_size, -1)
        gt = gt.contiguous().view(batch_size, -1).float()
        mask = mask.contiguous().view(batch_size, -1).float()

        pred = pred * mask
        gt = gt * mask

        a = torch.sum(pred * gt, dim=1)
        b = torch.sum(pred * pred, dim=1) + self.eps
        c = torch.sum(gt * gt, dim=1) + self.eps
        d = (2 * a) / (b + c)
        loss = 1 - d

        loss = torch.mean(loss)
        return loss
