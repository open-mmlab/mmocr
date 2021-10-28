# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.core import BitmapMasks

from mmocr.models.builder import LOSSES
from mmocr.utils import check_argument
from . import PANLoss


@LOSSES.register_module()
class PSELoss(PANLoss):
    """The class for implementing PSENet loss: Shape Robust Text Detection with
    Progressive Scale Expansion Network [https://arxiv.org/abs/1806.02559].

    This is partially adapted from https://github.com/whai362/PSENet.
    """

    def __init__(self,
                 alpha=0.7,
                 ohem_ratio=3,
                 reduction='mean',
                 kernel_sample_type='adaptive'):
        """Initialization.

        Args:
            alpha (float): alpha: The text loss coef;
                (1-alpha): the kernel loss coef.
            ohem_ratio (float): The negative/positive ratio in ohem.
            reduction (str): The way to reduce the loss.
        """
        super().__init__()
        assert reduction in ['mean',
                             'sum'], " reduction must in ['mean','sum']"
        self.alpha = alpha
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction
        self.kernel_sample_type = kernel_sample_type

    def forward(self, score_maps, downsample_ratio, gt_kernels, gt_mask):
        """Compute PSENet loss.

        Args:
            score_maps (tensor): The output tensor with size of Nx6xHxW.
            gt_kernels (list[BitmapMasks]): The kernel list with each element
                being the text kernel mask for one img.
            gt_mask (list[BitmapMasks]): The effective mask list
                with each element being the effective mask for one img.
            downsample_ratio (float): The downsample ratio between score_maps
                and the input img.

        Returns:
            results (dict): The loss.
        """

        assert check_argument.is_type_list(gt_kernels, BitmapMasks)
        assert check_argument.is_type_list(gt_mask, BitmapMasks)
        assert isinstance(downsample_ratio, float)
        losses = []

        pred_texts = score_maps[:, 0, :, :]
        pred_kernels = score_maps[:, 1:, :, :]
        feature_sz = score_maps.size()

        gt_kernels = [item.rescale(downsample_ratio) for item in gt_kernels]
        gt_kernels = self.bitmasks2tensor(gt_kernels, feature_sz[2:])
        gt_kernels = [item.to(score_maps.device) for item in gt_kernels]

        gt_mask = [item.rescale(downsample_ratio) for item in gt_mask]
        gt_mask = self.bitmasks2tensor(gt_mask, feature_sz[2:])
        gt_mask = [item.to(score_maps.device) for item in gt_mask]

        # compute text loss
        sampled_masks_text = self.ohem_batch(pred_texts.detach(),
                                             gt_kernels[0], gt_mask[0])
        loss_texts = self.dice_loss_with_logits(pred_texts, gt_kernels[0],
                                                sampled_masks_text)
        losses.append(self.alpha * loss_texts)

        # compute kernel loss
        if self.kernel_sample_type == 'hard':
            sampled_masks_kernel = (gt_kernels[0] > 0.5).float() * (
                gt_mask[0].float())
        elif self.kernel_sample_type == 'adaptive':
            sampled_masks_kernel = (pred_texts > 0).float() * (
                gt_mask[0].float())
        else:
            raise NotImplementedError

        num_kernel = pred_kernels.shape[1]
        assert num_kernel == len(gt_kernels) - 1
        loss_list = []
        for idx in range(num_kernel):
            loss_kernels = self.dice_loss_with_logits(
                pred_kernels[:, idx, :, :], gt_kernels[1 + idx],
                sampled_masks_kernel)
            loss_list.append(loss_kernels)

        losses.append((1 - self.alpha) * sum(loss_list) / len(loss_list))

        if self.reduction == 'mean':
            losses = [item.mean() for item in losses]
        elif self.reduction == 'sum':
            losses = [item.sum() for item in losses]
        else:
            raise NotImplementedError
        results = dict(loss_text=losses[0], loss_kernel=losses[1])
        return results
