import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES
from mmocr.models.common.losses import DiceLoss


@LOSSES.register_module()
class SegLoss(nn.Module):
    """Implementation of loss module for segmentation based text recognition
    method.

    Args:
        seg_downsample_ratio (float): Downsample ratio of
            segmentation map.
        seg_with_loss_weight (bool): If True, set weight for
            segmentation loss.
        ignore_index (int): Specifies a target value that is ignored
            and does not contribute to the input gradient.
    """

    def __init__(self,
                 seg_downsample_ratio=0.5,
                 seg_with_loss_weight=True,
                 ignore_index=255,
                 **kwargs):
        super().__init__()

        assert isinstance(seg_downsample_ratio, (int, float))
        assert 0 < seg_downsample_ratio <= 1
        assert isinstance(ignore_index, int)

        self.seg_downsample_ratio = seg_downsample_ratio
        self.seg_with_loss_weight = seg_with_loss_weight
        self.ignore_index = ignore_index

    def seg_loss(self, out_head, gt_kernels):
        seg_map = out_head  # bsz * num_classes * H/2 * W/2
        seg_target = [
            item[1].rescale(self.seg_downsample_ratio).to_tensor(
                torch.long, seg_map.device) for item in gt_kernels
        ]
        seg_target = torch.stack(seg_target).squeeze(1)

        loss_weight = None
        if self.seg_with_loss_weight:
            N = torch.sum(seg_target != self.ignore_index)
            N_neg = torch.sum(seg_target == 0)
            weight_val = 1.0 * N_neg / (N - N_neg)
            loss_weight = torch.ones(seg_map.size(1), device=seg_map.device)
            loss_weight[1:] = weight_val

        loss_seg = F.cross_entropy(
            seg_map,
            seg_target,
            weight=loss_weight,
            ignore_index=self.ignore_index)

        return loss_seg

    def forward(self, out_neck, out_head, gt_kernels):

        losses = {}

        loss_seg = self.seg_loss(out_head, gt_kernels)

        losses['loss_seg'] = loss_seg

        return losses


@LOSSES.register_module()
class CAFCNLoss(SegLoss):
    """Implementation of loss module in `CA-FCN.

    <https://arxiv.org/pdf/1809.06508.pdf>`_

    Args:
        alpha (float): Weight ratio of attention loss.
        attn_s2_downsample_ratio (float): Downsample ratio
            of attention map from output stage 2.
        attn_s3_downsample_ratio (float): Downsample ratio
            of attention map from output stage 3.
        seg_downsample_ratio (float): Downsample ratio of
            segmentation map.
        attn_with_dice_loss (bool): If True, use dice_loss for attention,
            else BCELoss.
        with_attn (bool): If True, include attention loss, else
            segmentation loss only.
        seg_with_loss_weight (bool): If True, set weight for
            segmentation loss.
        ignore_index (int): Specifies a target value that is ignored
            and does not contribute to the input gradient.
    """

    def __init__(self,
                 alpha=1.0,
                 attn_s2_downsample_ratio=0.25,
                 attn_s3_downsample_ratio=0.125,
                 seg_downsample_ratio=0.5,
                 attn_with_dice_loss=False,
                 with_attn=True,
                 seg_with_loss_weight=True,
                 ignore_index=255):
        super().__init__(seg_downsample_ratio, seg_with_loss_weight,
                         ignore_index)
        assert isinstance(alpha, (int, float))
        assert isinstance(attn_s2_downsample_ratio, (int, float))
        assert isinstance(attn_s3_downsample_ratio, (int, float))
        assert 0 < attn_s2_downsample_ratio <= 1
        assert 0 < attn_s3_downsample_ratio <= 1

        self.alpha = alpha
        self.attn_s2_downsample_ratio = attn_s2_downsample_ratio
        self.attn_s3_downsample_ratio = attn_s3_downsample_ratio
        self.with_attn = with_attn
        self.attn_with_dice_loss = attn_with_dice_loss

        # attention loss
        if with_attn:
            if attn_with_dice_loss:
                self.criterion_attn = DiceLoss()
            else:
                self.criterion_attn = nn.BCELoss(reduction='none')

    def attn_loss(self, out_neck, gt_kernels):
        attn_map_s2 = out_neck[0]  # bsz * 2 * H/4 * W/4

        mask_s2 = torch.stack([
            item[2].rescale(self.attn_s2_downsample_ratio).to_tensor(
                torch.float, attn_map_s2.device) for item in gt_kernels
        ])

        attn_target_s2 = torch.stack([
            item[0].rescale(self.attn_s2_downsample_ratio).to_tensor(
                torch.float, attn_map_s2.device) for item in gt_kernels
        ])

        mask_s3 = torch.stack([
            item[2].rescale(self.attn_s3_downsample_ratio).to_tensor(
                torch.float, attn_map_s2.device) for item in gt_kernels
        ])

        attn_target_s3 = torch.stack([
            item[0].rescale(self.attn_s3_downsample_ratio).to_tensor(
                torch.float, attn_map_s2.device) for item in gt_kernels
        ])

        targets = [
            attn_target_s2, attn_target_s3, attn_target_s3, attn_target_s3
        ]

        masks = [mask_s2, mask_s3, mask_s3, mask_s3]

        loss_attn = 0.
        for i in range(len(out_neck) - 1):
            pred = out_neck[i]
            if self.attn_with_dice_loss:
                loss_attn += self.criterion_attn(pred, targets[i], masks[i])
            else:
                loss_attn += torch.sum(
                    self.criterion_attn(pred, targets[i]) *
                    masks[i]) / torch.sum(masks[i])

        return loss_attn

    def forward(self, out_neck, out_head, gt_kernels):

        losses = super().forward(out_neck, out_head, gt_kernels)

        if self.with_attn:
            loss_attn = self.attn_loss(out_neck, gt_kernels)
            losses['loss_attn'] = loss_attn

        return losses
