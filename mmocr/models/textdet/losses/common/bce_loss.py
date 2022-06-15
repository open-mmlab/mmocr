# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch
import torch.nn as nn

from mmocr.registry import MODELS


@MODELS.register_module()
class MaskedBalancedBCELoss(nn.Module):
    """Masked Balanced BCE loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Defaults to 'none'.
        negative_ratio (float or int, optional): Maximum ratio of negative
            samples to positive ones. Defaults to 3.
        fallback_negative_num (int, optional): When the mask contains no
            positive samples, the number of negative samples to be sampled.
            Defaults to 0.
        eps (float, optional): Eps to avoid zero-division error.  Defaults to
            1e-6.
    """

    def __init__(self,
                 reduction: str = 'none',
                 negative_ratio: Union[float, int] = 3,
                 fallback_negative_num: int = 0,
                 eps: float = 1e-6) -> None:
        super().__init__()
        assert reduction in ['none', 'mean', 'sum']
        assert isinstance(negative_ratio, (float, int))
        assert isinstance(fallback_negative_num, int)
        assert isinstance(eps, float)
        self.eps = eps
        self.negative_ratio = negative_ratio
        self.reduction = reduction
        self.fallback_negative_num = fallback_negative_num
        self.binary_cross_entropy = nn.BCELoss(reduction=reduction)

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

        positive = (gt * mask).float()
        negative = ((1 - gt) * mask).float()
        positive_count = int(positive.sum())
        if positive_count == 0:
            negative_count = min(
                int(negative.sum()), self.fallback_negative_num)
        else:
            negative_count = min(
                int(negative.sum()), int(positive_count * self.negative_ratio))

        assert gt.max() <= 1 and gt.min() >= 0
        assert pred.max() <= 1 and pred.min() >= 0
        loss = self.binary_cross_entropy(pred, gt)
        positive_loss = loss * positive
        negative_loss = loss * negative

        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
            positive_count + negative_count + self.eps)

        return balance_loss


@MODELS.register_module()
class MaskedBCELoss(nn.Module):
    """Masked BCE loss.

    Args:
        eps (float, optional): Eps to avoid zero-division error.  Defaults to
            1e-6.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        assert isinstance(eps, float)
        self.eps = eps
        self.binary_cross_entropy = nn.BCELoss(reduction='none')

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

        assert gt.max() <= 1 and gt.min() >= 0
        assert pred.max() <= 1 and pred.min() >= 0
        loss = self.binary_cross_entropy(pred, gt)

        return (loss * mask) / (mask.sum() + self.eps)
