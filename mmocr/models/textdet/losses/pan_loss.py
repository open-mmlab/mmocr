# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from mmdet.core import BitmapMasks
from torch import nn

from mmocr.models.builder import LOSSES
from mmocr.utils import check_argument


@LOSSES.register_module()
class PANLoss(nn.Module):
    """The class for implementing PANet loss. This was partially adapted from
    https://github.com/WenmuZhou/PAN.pytorch.

    PANet: `Efficient and Accurate Arbitrary-
    Shaped Text Detection with Pixel Aggregation Network
    <https://arxiv.org/abs/1908.05900>`_.

    Args:
        alpha (float): The kernel loss coef.
        beta (float): The aggregation and discriminative loss coef.
        delta_aggregation (float): The constant for aggregation loss.
        delta_discrimination (float): The constant for discriminative loss.
        ohem_ratio (float): The negative/positive ratio in ohem.
        reduction (str): The way to reduce the loss.
        speedup_bbox_thr (int):  Speed up if speedup_bbox_thr > 0
            and < bbox num.
    """

    def __init__(self,
                 alpha=0.5,
                 beta=0.25,
                 delta_aggregation=0.5,
                 delta_discrimination=3,
                 ohem_ratio=3,
                 reduction='mean',
                 speedup_bbox_thr=-1):
        super().__init__()
        assert reduction in ['mean', 'sum'], "reduction must in ['mean','sum']"
        self.alpha = alpha
        self.beta = beta
        self.delta_aggregation = delta_aggregation
        self.delta_discrimination = delta_discrimination
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction
        self.speedup_bbox_thr = speedup_bbox_thr

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

    def forward(self, preds, downsample_ratio, gt_kernels, gt_mask):
        """Compute PANet loss.

        Args:
            preds (Tensor): The output tensor of size :math:`(N, 6, H, W)`.
            downsample_ratio (float): The downsample ratio between preds
                and the input img.
            gt_kernels (list[BitmapMasks]): The kernel list with each element
                being the text kernel mask for one img.
            gt_mask (list[BitmapMasks]): The effective mask list
                with each element being the effective mask for one img.

        Returns:
            dict:  A loss dict with ``loss_text``, ``loss_kernel``,
            ``loss_aggregation`` and ``loss_discrimination``.
        """

        assert check_argument.is_type_list(gt_kernels, BitmapMasks)
        assert check_argument.is_type_list(gt_mask, BitmapMasks)
        assert isinstance(downsample_ratio, float)

        pred_texts = preds[:, 0, :, :]
        pred_kernels = preds[:, 1, :, :]
        inst_embed = preds[:, 2:, :, :]
        feature_sz = preds.size()

        mapping = {'gt_kernels': gt_kernels, 'gt_mask': gt_mask}
        gt = {}
        for key, value in mapping.items():
            gt[key] = value
            gt[key] = [item.rescale(downsample_ratio) for item in gt[key]]
            gt[key] = self.bitmasks2tensor(gt[key], feature_sz[2:])
            gt[key] = [item.to(preds.device) for item in gt[key]]
        loss_aggrs, loss_discrs = self.aggregation_discrimination_loss(
            gt['gt_kernels'][0], gt['gt_kernels'][1], inst_embed)
        # compute text loss
        sampled_mask = self.ohem_batch(pred_texts.detach(),
                                       gt['gt_kernels'][0], gt['gt_mask'][0])
        loss_texts = self.dice_loss_with_logits(pred_texts,
                                                gt['gt_kernels'][0],
                                                sampled_mask)

        # compute kernel loss

        sampled_masks_kernel = (gt['gt_kernels'][0] > 0.5).float() * (
            gt['gt_mask'][0].float())
        loss_kernels = self.dice_loss_with_logits(pred_kernels,
                                                  gt['gt_kernels'][1],
                                                  sampled_masks_kernel)
        losses = [loss_texts, loss_kernels, loss_aggrs, loss_discrs]
        if self.reduction == 'mean':
            losses = [item.mean() for item in losses]
        elif self.reduction == 'sum':
            losses = [item.sum() for item in losses]
        else:
            raise NotImplementedError

        coefs = [1, self.alpha, self.beta, self.beta]
        losses = [item * scale for item, scale in zip(losses, coefs)]

        results = dict()
        results.update(
            loss_text=losses[0],
            loss_kernel=losses[1],
            loss_aggregation=losses[2],
            loss_discrimination=losses[3])
        return results

    def aggregation_discrimination_loss(self, gt_texts, gt_kernels,
                                        inst_embeds):
        """Compute the aggregation and discrimnative losses.

        Args:
            gt_texts (Tensor): The ground truth text mask of size
                :math:`(N, 1, H, W)`.
            gt_kernels (Tensor): The ground truth text kernel mask of
                size :math:`(N, 1, H, W)`.
            inst_embeds(Tensor): The text instance embedding tensor
                of size :math:`(N, 1, H, W)`.

        Returns:
            (Tensor, Tensor): A tuple of aggregation loss and discriminative
            loss before reduction.
        """

        batch_size = gt_texts.size()[0]
        gt_texts = gt_texts.contiguous().reshape(batch_size, -1)
        gt_kernels = gt_kernels.contiguous().reshape(batch_size, -1)

        assert inst_embeds.shape[1] == 4
        inst_embeds = inst_embeds.contiguous().reshape(batch_size, 4, -1)

        loss_aggrs = []
        loss_discrs = []

        for text, kernel, embed in zip(gt_texts, gt_kernels, inst_embeds):

            # for each image
            text_num = int(text.max().item())
            loss_aggr_img = []
            kernel_avgs = []
            select_num = self.speedup_bbox_thr
            if 0 < select_num < text_num:
                inds = np.random.choice(
                    text_num, select_num, replace=False) + 1
            else:
                inds = range(1, text_num + 1)

            for i in inds:
                # for each text instance
                kernel_i = (kernel == i)  # 0.2ms
                if kernel_i.sum() == 0 or (text == i).sum() == 0:  # 0.2ms
                    continue

                # compute G_Ki in Eq (2)
                avg = embed[:, kernel_i].mean(1)  # 0.5ms
                kernel_avgs.append(avg)

                embed_i = embed[:, text == i]  # 0.6ms
                # ||F(p) - G(K_i)|| - delta_aggregation, shape: nums
                distance = (embed_i - avg.reshape(4, 1)).norm(  # 0.5ms
                    2, dim=0) - self.delta_aggregation
                # compute D(p,K_i) in Eq (2)
                hinge = torch.max(
                    distance,
                    torch.tensor(0, device=distance.device,
                                 dtype=torch.float)).pow(2)

                aggr = torch.log(hinge + 1).mean()
                loss_aggr_img.append(aggr)

            num_inst = len(loss_aggr_img)
            if num_inst > 0:
                loss_aggr_img = torch.stack(loss_aggr_img).mean()
            else:
                loss_aggr_img = torch.tensor(
                    0, device=gt_texts.device, dtype=torch.float)
            loss_aggrs.append(loss_aggr_img)

            loss_discr_img = 0
            for avg_i, avg_j in itertools.combinations(kernel_avgs, 2):
                # delta_discrimination - ||G(K_i) - G(K_j)||
                distance_ij = self.delta_discrimination - (avg_i -
                                                           avg_j).norm(2)
                # D(K_i,K_j)
                D_ij = torch.max(
                    distance_ij,
                    torch.tensor(
                        0, device=distance_ij.device,
                        dtype=torch.float)).pow(2)
                loss_discr_img += torch.log(D_ij + 1)

            if num_inst > 1:
                loss_discr_img /= (num_inst * (num_inst - 1))
            else:
                loss_discr_img = torch.tensor(
                    0, device=gt_texts.device, dtype=torch.float)
            if num_inst == 0:
                warnings.warn('num of instance is 0')
            loss_discrs.append(loss_discr_img)
        return torch.stack(loss_aggrs), torch.stack(loss_discrs)

    def dice_loss_with_logits(self, pred, target, mask):

        smooth = 0.001

        pred = torch.sigmoid(pred)
        target[target <= 0.5] = 0
        target[target > 0.5] = 1
        pred = pred.contiguous().view(pred.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1)
        mask = mask.contiguous().view(mask.size()[0], -1)

        pred = pred * mask
        target = target * mask

        a = torch.sum(pred * target, 1) + smooth
        b = torch.sum(pred * pred, 1) + smooth
        c = torch.sum(target * target, 1) + smooth
        d = (2 * a) / (b + c)
        return 1 - d

    def ohem_img(self, text_score, gt_text, gt_mask):
        """Sample the top-k maximal negative samples and all positive samples.

        Args:
            text_score (Tensor): The text score of size :math:`(H, W)`.
            gt_text (Tensor): The ground truth text mask of size
                :math:`(H, W)`.
            gt_mask (Tensor): The effective region mask of size :math:`(H, W)`.

        Returns:
            Tensor: The sampled pixel mask of size :math:`(H, W)`.
        """
        assert isinstance(text_score, torch.Tensor)
        assert isinstance(gt_text, torch.Tensor)
        assert isinstance(gt_mask, torch.Tensor)
        assert len(text_score.shape) == 2
        assert text_score.shape == gt_text.shape
        assert gt_text.shape == gt_mask.shape

        pos_num = (int)(torch.sum(gt_text > 0.5).item()) - (int)(
            torch.sum((gt_text > 0.5) * (gt_mask <= 0.5)).item())
        neg_num = (int)(torch.sum(gt_text <= 0.5).item())
        neg_num = (int)(min(pos_num * self.ohem_ratio, neg_num))

        if pos_num == 0 or neg_num == 0:
            warnings.warn('pos_num = 0 or neg_num = 0')
            return gt_mask.bool()

        neg_score = text_score[gt_text <= 0.5]
        neg_score_sorted, _ = torch.sort(neg_score, descending=True)
        threshold = neg_score_sorted[neg_num - 1]
        sampled_mask = (((text_score >= threshold) + (gt_text > 0.5)) > 0) * (
            gt_mask > 0.5)
        return sampled_mask

    def ohem_batch(self, text_scores, gt_texts, gt_mask):
        """OHEM sampling for a batch of imgs.

        Args:
            text_scores (Tensor): The text scores of size :math:`(H, W)`.
            gt_texts (Tensor): The gt text masks of size :math:`(H, W)`.
            gt_mask (Tensor): The gt effective mask of size :math:`(H, W)`.

        Returns:
            Tensor: The sampled mask of size :math:`(H, W)`.
        """
        assert isinstance(text_scores, torch.Tensor)
        assert isinstance(gt_texts, torch.Tensor)
        assert isinstance(gt_mask, torch.Tensor)
        assert len(text_scores.shape) == 3
        assert text_scores.shape == gt_texts.shape
        assert gt_texts.shape == gt_mask.shape

        sampled_masks = []
        for i in range(text_scores.shape[0]):
            sampled_masks.append(
                self.ohem_img(text_scores[i], gt_texts[i], gt_mask[i]))

        sampled_masks = torch.stack(sampled_masks)

        return sampled_masks
