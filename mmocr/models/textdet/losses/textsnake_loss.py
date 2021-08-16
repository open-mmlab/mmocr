# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmdet.core import BitmapMasks
from mmdet.models.builder import LOSSES
from torch import nn

from mmocr.utils import check_argument


@LOSSES.register_module()
class TextSnakeLoss(nn.Module):
    """The class for implementing TextSnake loss:
    TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes
    [https://arxiv.org/abs/1807.01544].
    This is partially adapted from
    https://github.com/princewang1994/TextSnake.pytorch.
    """

    def __init__(self, ohem_ratio=3.0):
        """Initialization.

        Args:
            ohem_ratio (float): The negative/positive ratio in ohem.
        """
        super().__init__()
        self.ohem_ratio = ohem_ratio

    def balanced_bce_loss(self, pred, gt, mask):

        assert pred.shape == gt.shape == mask.shape
        positive = gt * mask
        negative = (1 - gt) * mask
        positive_count = int(positive.float().sum())
        gt = gt.float()
        if positive_count > 0:
            loss = F.binary_cross_entropy(pred, gt, reduction='none')
            positive_loss = torch.sum(loss * positive.float())
            negative_loss = loss * negative.float()
            negative_count = min(
                int(negative.float().sum()),
                int(positive_count * self.ohem_ratio))
        else:
            positive_loss = torch.tensor(0.0, device=pred.device)
            loss = F.binary_cross_entropy(pred, gt, reduction='none')
            negative_loss = loss * negative.float()
            negative_count = 100
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = (positive_loss + torch.sum(negative_loss)) / (
            float(positive_count + negative_count) + 1e-5)

        return balance_loss

    def bitmasks2tensor(self, bitmasks, target_sz):
        """Convert Bitmasks to tensor.

        Args:
            bitmasks (list[BitmapMasks]): The BitmapMasks list. Each item is
                for one img.
            target_sz (tuple(int, int)): The target tensor size HxW.

        Returns
            results (list[tensor]): The list of kernel tensors. Each
                element is for one kernel level.
        """
        assert check_argument.is_type_list(bitmasks, BitmapMasks)
        assert isinstance(target_sz, tuple)

        batch_size = len(bitmasks)
        num_masks = len(bitmasks[0])

        results = []

        for level_inx in range(num_masks):
            kernel = []
            for batch_inx in range(batch_size):
                mask = torch.from_numpy(bitmasks[batch_inx].masks[level_inx])
                # hxw
                mask_sz = mask.shape
                # left, right, top, bottom
                pad = [
                    0, target_sz[1] - mask_sz[1], 0, target_sz[0] - mask_sz[0]
                ]
                mask = F.pad(mask, pad, mode='constant', value=0)
                kernel.append(mask)
            kernel = torch.stack(kernel)
            results.append(kernel)

        return results

    def forward(self, pred_maps, downsample_ratio, gt_text_mask,
                gt_center_region_mask, gt_mask, gt_radius_map, gt_sin_map,
                gt_cos_map):

        assert isinstance(downsample_ratio, float)
        assert check_argument.is_type_list(gt_text_mask, BitmapMasks)
        assert check_argument.is_type_list(gt_center_region_mask, BitmapMasks)
        assert check_argument.is_type_list(gt_mask, BitmapMasks)
        assert check_argument.is_type_list(gt_radius_map, BitmapMasks)
        assert check_argument.is_type_list(gt_sin_map, BitmapMasks)
        assert check_argument.is_type_list(gt_cos_map, BitmapMasks)

        pred_text_region = pred_maps[:, 0, :, :]
        pred_center_region = pred_maps[:, 1, :, :]
        pred_sin_map = pred_maps[:, 2, :, :]
        pred_cos_map = pred_maps[:, 3, :, :]
        pred_radius_map = pred_maps[:, 4, :, :]
        feature_sz = pred_maps.size()
        device = pred_maps.device

        # bitmask 2 tensor
        mapping = {
            'gt_text_mask': gt_text_mask,
            'gt_center_region_mask': gt_center_region_mask,
            'gt_mask': gt_mask,
            'gt_radius_map': gt_radius_map,
            'gt_sin_map': gt_sin_map,
            'gt_cos_map': gt_cos_map
        }
        gt = {}
        for key, value in mapping.items():
            gt[key] = value
            if abs(downsample_ratio - 1.0) < 1e-2:
                gt[key] = self.bitmasks2tensor(gt[key], feature_sz[2:])
            else:
                gt[key] = [item.rescale(downsample_ratio) for item in gt[key]]
                gt[key] = self.bitmasks2tensor(gt[key], feature_sz[2:])
                if key == 'gt_radius_map':
                    gt[key] = [item * downsample_ratio for item in gt[key]]
            gt[key] = [item.to(device) for item in gt[key]]

        scale = torch.sqrt(1.0 / (pred_sin_map**2 + pred_cos_map**2 + 1e-8))
        pred_sin_map = pred_sin_map * scale
        pred_cos_map = pred_cos_map * scale

        loss_text = self.balanced_bce_loss(
            torch.sigmoid(pred_text_region), gt['gt_text_mask'][0],
            gt['gt_mask'][0])

        text_mask = (gt['gt_text_mask'][0] * gt['gt_mask'][0]).float()
        loss_center_map = F.binary_cross_entropy(
            torch.sigmoid(pred_center_region),
            gt['gt_center_region_mask'][0].float(),
            reduction='none')
        if int(text_mask.sum()) > 0:
            loss_center = torch.sum(
                loss_center_map * text_mask) / torch.sum(text_mask)
        else:
            loss_center = torch.tensor(0.0, device=device)

        center_mask = (gt['gt_center_region_mask'][0] *
                       gt['gt_mask'][0]).float()
        if int(center_mask.sum()) > 0:
            map_sz = pred_radius_map.size()
            ones = torch.ones(map_sz, dtype=torch.float, device=device)
            loss_radius = torch.sum(
                F.smooth_l1_loss(
                    pred_radius_map / (gt['gt_radius_map'][0] + 1e-2),
                    ones,
                    reduction='none') * center_mask) / torch.sum(center_mask)
            loss_sin = torch.sum(
                F.smooth_l1_loss(
                    pred_sin_map, gt['gt_sin_map'][0], reduction='none') *
                center_mask) / torch.sum(center_mask)
            loss_cos = torch.sum(
                F.smooth_l1_loss(
                    pred_cos_map, gt['gt_cos_map'][0], reduction='none') *
                center_mask) / torch.sum(center_mask)
        else:
            loss_radius = torch.tensor(0.0, device=device)
            loss_sin = torch.tensor(0.0, device=device)
            loss_cos = torch.tensor(0.0, device=device)

        results = dict(
            loss_text=loss_text,
            loss_center=loss_center,
            loss_radius=loss_radius,
            loss_sin=loss_sin,
            loss_cos=loss_cos)

        return results
