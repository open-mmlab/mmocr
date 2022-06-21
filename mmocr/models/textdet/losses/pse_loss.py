# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from mmocr.core import TextDetDataSample
from mmocr.registry import MODELS
from . import PANLoss


@MODELS.register_module()
class PSELoss(PANLoss):
    """The class for implementing PSENet loss. This is partially adapted from
    https://github.com/whai362/PSENet.

    PSENet: `Shape Robust Text Detection with
    Progressive Scale Expansion Network <https://arxiv.org/abs/1806.02559>`_.

    Args:
        weight_text (float): The weight of text loss. Defaults to 0.7.
        weight_kernel (float): The weight of text kernel. Defaults to 0.3.
        loss_text (dict): Loss type for text. Defaults to
            dict('MaskedSquareDiceLoss').
        loss_kernel (dict): Loss type for kernel. Defaults to
            dict('MaskedSquareDiceLoss').
        ohem_ratio (int or float): The negative/positive ratio in ohem.
            Defaults to 3.
        reduction (str): The way to reduce the loss. Defaults to 'mean'.
            Options are 'mean' and 'sum'.
        kernel_sample_type (str): The way to sample kernel. Defaults to
            adaptive. Options are 'adaptive' and 'hard'.
        shrink_ratio (tuple): The ratio for shirinking text instances.
            Defaults to (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4).
        max_shrink_dist (int or float): The maximum shrinking distance.
            Defaults to 20.
    """

    def __init__(
        self,
        weight_text: float = 0.7,
        weight_kernel: float = 0.3,
        loss_text: Dict = dict(type='MaskedSquareDiceLoss'),
        loss_kernel: Dict = dict(type='MaskedSquareDiceLoss'),
        ohem_ratio: Union[int, float] = 3,
        reduction: str = 'mean',
        kernel_sample_type: str = 'adaptive',
        shrink_ratio: Tuple[float] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4),
        max_shrink_dist: Union[int, float] = 20,
    ) -> None:
        super().__init__()
        assert reduction in ['mean', 'sum'
                             ], "reduction must be either of ['mean','sum']"
        assert kernel_sample_type in [
            'adaptive', 'hard'
        ], "kernel_sample_type must be either of ['hard', 'adaptive']"
        self.weight_text = weight_text
        self.weight_kernel = weight_kernel
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction
        self.shrink_ratio = shrink_ratio
        self.kernel_sample_type = kernel_sample_type
        self.max_shrink_dist = max_shrink_dist
        self.loss_text = MODELS.build(loss_text)
        self.loss_kernel = MODELS.build(loss_kernel)

    def forward(self, preds: torch.Tensor,
                data_samples: Sequence[TextDetDataSample]) -> Dict:
        """Compute PSENet loss.

        Args:
            preds (torch.Tensor): Raw predictions from model with
                shape :math:`(N, C, H, W)`.
            data_samples (list[TextDetDataSample]): The data samples.

        Returns:
            dict: The dict for pse losses with loss_text, loss_kernel,
            loss_aggregation and loss_discrimination.
        """
        losses = []

        gt_kernels, gt_masks = self.get_targets(data_samples)
        target_size = gt_kernels.size()[2:]
        preds = F.interpolate(preds, size=target_size, mode='bilinear')
        pred_texts = preds[:, 0, :, :]
        pred_kernels = preds[:, 1:, :, :]

        gt_kernels = gt_kernels.to(preds.device)
        gt_kernels[gt_kernels <= 0.5] = 0
        gt_kernels[gt_kernels > 0.5] = 1
        gt_masks = gt_masks.to(preds.device)

        # compute text loss
        sampled_mask = self._ohem_batch(pred_texts.detach(), gt_kernels[0],
                                        gt_masks)
        loss_texts = self.loss_text(pred_texts.sigmoid(), gt_kernels[0],
                                    sampled_mask)
        losses.append(self.weight_text * loss_texts)

        # compute kernel loss
        if self.kernel_sample_type == 'hard':
            sampled_masks_kernel = (gt_kernels[0] >
                                    0.5).float() * gt_masks.float()
        elif self.kernel_sample_type == 'adaptive':
            sampled_masks_kernel = (pred_texts > 0).float() * (
                gt_masks.float())
        else:
            raise NotImplementedError

        num_kernel = pred_kernels.shape[1]
        assert num_kernel == len(gt_kernels) - 1
        loss_list = []
        for idx in range(num_kernel):
            loss_kernels = self.loss_kernel(
                pred_kernels[:, idx, :, :].sigmoid(), gt_kernels[1 + idx],
                sampled_masks_kernel)
            loss_list.append(loss_kernels)

        losses.append(self.weight_kernel * sum(loss_list) / len(loss_list))

        if self.reduction == 'mean':
            losses = [item.mean() for item in losses]
        elif self.reduction == 'sum':
            losses = [item.sum() for item in losses]
        else:
            raise NotImplementedError

        results = dict(loss_text=losses[0], loss_kernel=losses[1])
        return results
