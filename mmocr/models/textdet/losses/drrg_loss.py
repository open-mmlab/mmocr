# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmdet.core import BitmapMasks
from torch import nn

from mmocr.models.builder import LOSSES
from mmocr.utils import check_argument


@LOSSES.register_module()
class DRRGLoss(nn.Module):
    """The class for implementing DRRG loss. This is partially adapted from
    https://github.com/GXYM/DRRG licensed under the MIT license.

    DRRG: `Deep Relational Reasoning Graph Network for Arbitrary Shape Text
    Detection <https://arxiv.org/abs/1908.05900>`_.

    Args:
        ohem_ratio (float): The negative/positive ratio in ohem.
    """

    def __init__(self, ohem_ratio=3.0):
        super().__init__()
        self.ohem_ratio = ohem_ratio

    def balance_bce_loss(self, pred, gt, mask):
        """Balanced Binary-CrossEntropy Loss.

        Args:
            pred (Tensor): Shape of :math:`(1, H, W)`.
            gt (Tensor): Shape of :math:`(1, H, W)`.
            mask (Tensor): Shape of :math:`(1, H, W)`.

        Returns:
            Tensor: Balanced bce loss.
        """
        assert pred.shape == gt.shape == mask.shape
        assert torch.all(pred >= 0) and torch.all(pred <= 1)
        assert torch.all(gt >= 0) and torch.all(gt <= 1)
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

    def gcn_loss(self, gcn_data):
        """CrossEntropy Loss from gcn module.

        Args:
            gcn_data (tuple(Tensor, Tensor)): The first is the
                prediction with shape :math:`(N, 2)` and the
                second is the gt label with shape :math:`(m, n)`
                where :math:`m * n = N`.

        Returns:
            Tensor: CrossEntropy loss.
        """
        gcn_pred, gt_labels = gcn_data
        gt_labels = gt_labels.view(-1).to(gcn_pred.device)
        loss = F.cross_entropy(gcn_pred, gt_labels)

        return loss

    def bitmasks2tensor(self, bitmasks, target_sz):
        """Convert Bitmasks to tensor.

        Args:
            bitmasks (list[BitmapMasks]): The BitmapMasks list. Each item is
                for one img.
            target_sz (tuple(int, int)): The target tensor of size
                :math:`(H, W)`.

        Returns:
            list[Tensor]: The list of kernel tensors. Each element stands for
            one kernel level.
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

    def forward(self, preds, downsample_ratio, gt_text_mask,
                gt_center_region_mask, gt_mask, gt_top_height_map,
                gt_bot_height_map, gt_sin_map, gt_cos_map):
        """Compute Drrg loss.

        Args:
            preds (tuple(Tensor)): The first is the prediction map
                with shape :math:`(N, C_{out}, H, W)`.
                The second is prediction from GCN module, with
                shape :math:`(N, 2)`.
                The third is ground-truth label with shape :math:`(N, 8)`.
            downsample_ratio (float): The downsample ratio.
            gt_text_mask (list[BitmapMasks]): Text mask.
            gt_center_region_mask (list[BitmapMasks]): Center region mask.
            gt_mask (list[BitmapMasks]): Effective mask.
            gt_top_height_map (list[BitmapMasks]): Top height map.
            gt_bot_height_map (list[BitmapMasks]): Bottom height map.
            gt_sin_map (list[BitmapMasks]): Sinusoid map.
            gt_cos_map (list[BitmapMasks]): Cosine map.

        Returns:
            dict:  A loss dict with ``loss_text``, ``loss_center``,
            ``loss_height``, ``loss_sin``, ``loss_cos``, and ``loss_gcn``.
        """
        assert isinstance(preds, tuple)
        assert isinstance(downsample_ratio, float)
        assert check_argument.is_type_list(gt_text_mask, BitmapMasks)
        assert check_argument.is_type_list(gt_center_region_mask, BitmapMasks)
        assert check_argument.is_type_list(gt_mask, BitmapMasks)
        assert check_argument.is_type_list(gt_top_height_map, BitmapMasks)
        assert check_argument.is_type_list(gt_bot_height_map, BitmapMasks)
        assert check_argument.is_type_list(gt_sin_map, BitmapMasks)
        assert check_argument.is_type_list(gt_cos_map, BitmapMasks)

        pred_maps, gcn_data = preds
        pred_text_region = pred_maps[:, 0, :, :]
        pred_center_region = pred_maps[:, 1, :, :]
        pred_sin_map = pred_maps[:, 2, :, :]
        pred_cos_map = pred_maps[:, 3, :, :]
        pred_top_height_map = pred_maps[:, 4, :, :]
        pred_bot_height_map = pred_maps[:, 5, :, :]
        feature_sz = pred_maps.size()
        device = pred_maps.device

        # bitmask 2 tensor
        mapping = {
            'gt_text_mask': gt_text_mask,
            'gt_center_region_mask': gt_center_region_mask,
            'gt_mask': gt_mask,
            'gt_top_height_map': gt_top_height_map,
            'gt_bot_height_map': gt_bot_height_map,
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
                if key in ['gt_top_height_map', 'gt_bot_height_map']:
                    gt[key] = [item * downsample_ratio for item in gt[key]]
            gt[key] = [item.to(device) for item in gt[key]]

        scale = torch.sqrt(1.0 / (pred_sin_map**2 + pred_cos_map**2 + 1e-8))
        pred_sin_map = pred_sin_map * scale
        pred_cos_map = pred_cos_map * scale

        loss_text = self.balance_bce_loss(
            torch.sigmoid(pred_text_region), gt['gt_text_mask'][0],
            gt['gt_mask'][0])

        text_mask = (gt['gt_text_mask'][0] * gt['gt_mask'][0]).float()
        negative_text_mask = ((1 - gt['gt_text_mask'][0]) *
                              gt['gt_mask'][0]).float()
        loss_center_map = F.binary_cross_entropy(
            torch.sigmoid(pred_center_region),
            gt['gt_center_region_mask'][0].float(),
            reduction='none')
        if int(text_mask.sum()) > 0:
            loss_center_positive = torch.sum(
                loss_center_map * text_mask) / torch.sum(text_mask)
        else:
            loss_center_positive = torch.tensor(0.0, device=device)
        loss_center_negative = torch.sum(
            loss_center_map *
            negative_text_mask) / torch.sum(negative_text_mask)
        loss_center = loss_center_positive + 0.5 * loss_center_negative

        center_mask = (gt['gt_center_region_mask'][0] *
                       gt['gt_mask'][0]).float()
        if int(center_mask.sum()) > 0:
            map_sz = pred_top_height_map.size()
            ones = torch.ones(map_sz, dtype=torch.float, device=device)
            loss_top = F.smooth_l1_loss(
                pred_top_height_map / (gt['gt_top_height_map'][0] + 1e-2),
                ones,
                reduction='none')
            loss_bot = F.smooth_l1_loss(
                pred_bot_height_map / (gt['gt_bot_height_map'][0] + 1e-2),
                ones,
                reduction='none')
            gt_height = (
                gt['gt_top_height_map'][0] + gt['gt_bot_height_map'][0])
            loss_height = torch.sum(
                (torch.log(gt_height + 1) *
                 (loss_top + loss_bot)) * center_mask) / torch.sum(center_mask)

            loss_sin = torch.sum(
                F.smooth_l1_loss(
                    pred_sin_map, gt['gt_sin_map'][0], reduction='none') *
                center_mask) / torch.sum(center_mask)
            loss_cos = torch.sum(
                F.smooth_l1_loss(
                    pred_cos_map, gt['gt_cos_map'][0], reduction='none') *
                center_mask) / torch.sum(center_mask)
        else:
            loss_height = torch.tensor(0.0, device=device)
            loss_sin = torch.tensor(0.0, device=device)
            loss_cos = torch.tensor(0.0, device=device)

        loss_gcn = self.gcn_loss(gcn_data)

        results = dict(
            loss_text=loss_text,
            loss_center=loss_center,
            loss_height=loss_height,
            loss_sin=loss_sin,
            loss_cos=loss_cos,
            loss_gcn=loss_gcn)

        return results
