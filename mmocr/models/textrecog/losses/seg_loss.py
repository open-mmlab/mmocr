import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES


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
