# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmdet.models.utils import multi_apply
from torch import nn

from mmocr.registry import MODELS
from mmocr.structures import TextDetDataSample
from .seg_based_module_loss import SegBasedModuleLoss


@MODELS.register_module()
class PANModuleLoss(SegBasedModuleLoss):
    """The class for implementing PANet loss. This was partially adapted from
    https://github.com/whai362/pan_pp.pytorch and
    https://github.com/WenmuZhou/PAN.pytorch.

    PANet: `Efficient and Accurate Arbitrary-
    Shaped Text Detection with Pixel Aggregation Network
    <https://arxiv.org/abs/1908.05900>`_.

    Args:
        loss_text (dict) The loss config for text map. Defaults to
            dict(type='MaskedSquareDiceLoss').
        loss_kernel (dict) The loss config for kernel map. Defaults to
            dict(type='MaskedSquareDiceLoss').
        loss_embedding (dict) The loss config for embedding map. Defaults to
            dict(type='PANEmbLossV1').
        weight_text (float): The weight of text loss. Defaults to 1.
        weight_kernel (float): The weight of kernel loss. Defaults to 0.5.
        weight_embedding (float): The weight of embedding loss.
            Defaults to 0.25.
        ohem_ratio (float): The negative/positive ratio in ohem. Defaults to 3.
        shrink_ratio (tuple[float]) : The ratio of shrinking kernel. Defaults
            to (1.0, 0.5).
        max_shrink_dist (int or float): The maximum shrinking distance.
            Defaults to 20.
        reduction (str): The way to reduce the loss. Available options are
            "mean" and "sum". Defaults to 'mean'.
    """

    def __init__(
            self,
            loss_text: Dict = dict(type='MaskedSquareDiceLoss'),
            loss_kernel: Dict = dict(type='MaskedSquareDiceLoss'),
            loss_embedding: Dict = dict(type='PANEmbLossV1'),
            weight_text: float = 1.0,
            weight_kernel: float = 0.5,
            weight_embedding: float = 0.25,
            ohem_ratio: Union[int, float] = 3,  # TODO Find a better name
            shrink_ratio: Sequence[Union[int, float]] = (1.0, 0.5),
            max_shrink_dist: Union[int, float] = 20,
            reduction: str = 'mean') -> None:
        super().__init__()
        assert reduction in ['mean', 'sum'], "reduction must in ['mean','sum']"
        self.weight_text = weight_text
        self.weight_kernel = weight_kernel
        self.weight_embedding = weight_embedding
        self.shrink_ratio = shrink_ratio
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction
        self.max_shrink_dist = max_shrink_dist
        self.loss_text = MODELS.build(loss_text)
        self.loss_kernel = MODELS.build(loss_kernel)
        self.loss_embedding = MODELS.build(loss_embedding)

    def forward(self, preds: torch.Tensor,
                data_samples: Sequence[TextDetDataSample]) -> Dict:
        """Compute PAN loss.

        Args:
            preds (dict): Raw predictions from model with
                shape :math:`(N, C, H, W)`.
            data_samples (list[TextDetDataSample]): The data samples.

        Returns:
            dict: The dict for pan losses with loss_text, loss_kernel,
            loss_aggregation and loss_discrimination.
        """

        gt_kernels, gt_masks = self.get_targets(data_samples)
        target_size = gt_kernels.size()[2:]
        preds = F.interpolate(preds, size=target_size, mode='bilinear')
        pred_texts = preds[:, 0, :, :]
        pred_kernels = preds[:, 1, :, :]
        inst_embed = preds[:, 2:, :, :]
        gt_kernels = gt_kernels.to(preds.device)
        gt_masks = gt_masks.to(preds.device)

        # compute embedding loss
        loss_emb = self.loss_embedding(inst_embed, gt_kernels[0],
                                       gt_kernels[1], gt_masks)
        gt_kernels[gt_kernels <= 0.5] = 0
        gt_kernels[gt_kernels > 0.5] = 1
        # compute text loss
        sampled_mask = self._ohem_batch(pred_texts.detach(), gt_kernels[0],
                                        gt_masks)
        pred_texts = torch.sigmoid(pred_texts)
        loss_texts = self.loss_text(pred_texts, gt_kernels[0], sampled_mask)

        # compute kernel loss
        pred_kernels = torch.sigmoid(pred_kernels)
        sampled_masks_kernel = (gt_kernels[0] > 0.5).float() * gt_masks
        loss_kernels = self.loss_kernel(pred_kernels, gt_kernels[1],
                                        sampled_masks_kernel)

        losses = [loss_texts, loss_kernels, loss_emb]
        if self.reduction == 'mean':
            losses = [item.mean() for item in losses]
        else:
            losses = [item.sum() for item in losses]

        results = dict()
        results.update(
            loss_text=self.weight_text * losses[0],
            loss_kernel=self.weight_kernel * losses[1],
            loss_embedding=self.weight_embedding * losses[2])
        return results

    def get_targets(
        self,
        data_samples: Sequence[TextDetDataSample],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate the gt targets for PANet.

        Args:
            results (dict): The input result dictionary.

        Returns:
            results (dict): The output result dictionary.
        """
        gt_kernels, gt_masks = multi_apply(self._get_target_single,
                                           data_samples)
        # gt_kernels: (N, kernel_number, H, W)->(kernel_number, N, H, W)
        gt_kernels = torch.stack(gt_kernels, dim=0).permute(1, 0, 2, 3)
        gt_masks = torch.stack(gt_masks, dim=0)
        return gt_kernels, gt_masks

    def _get_target_single(self, data_sample: TextDetDataSample
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate loss target from a data sample.

        Args:
            data_sample (TextDetDataSample): The data sample.

        Returns:
            tuple: A tuple of four tensors as the targets of one prediction.
        """
        gt_polygons = data_sample.gt_instances.polygons
        gt_ignored = data_sample.gt_instances.ignored

        gt_kernels = []
        for ratio in self.shrink_ratio:
            # TODO pass `gt_ignored` to `_generate_kernels`
            gt_kernel, _ = self._generate_kernels(
                data_sample.batch_input_shape,
                gt_polygons,
                ratio,
                ignore_flags=None,
                max_shrink_dist=self.max_shrink_dist)
            gt_kernels.append(gt_kernel)
        gt_polygons_ignored = data_sample.gt_instances[gt_ignored].polygons
        gt_mask = self._generate_effective_mask(data_sample.batch_input_shape,
                                                gt_polygons_ignored)

        gt_kernels = np.stack(gt_kernels, axis=0)
        gt_kernels = torch.from_numpy(gt_kernels).float()
        gt_mask = torch.from_numpy(gt_mask).float()
        return gt_kernels, gt_mask

    def _ohem_batch(self, text_scores: torch.Tensor, gt_texts: torch.Tensor,
                    gt_mask: torch.Tensor) -> torch.Tensor:
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
                self._ohem_single(text_scores[i], gt_texts[i], gt_mask[i]))

        sampled_masks = torch.stack(sampled_masks)

        return sampled_masks

    def _ohem_single(self, text_score: torch.Tensor, gt_text: torch.Tensor,
                     gt_mask: torch.Tensor) -> torch.Tensor:
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


@MODELS.register_module()
class PANEmbLossV1(nn.Module):
    """The class for implementing EmbLossV1. This was partially adapted from
    https://github.com/whai362/pan_pp.pytorch.

    Args:
        feature_dim (int): The dimension of the feature. Defaults to 4.
        delta_aggregation (float): The delta for aggregation. Defaults to 0.5.
        delta_discrimination (float): The delta for discrimination.
            Defaults to 1.5.
    """

    def __init__(self,
                 feature_dim: int = 4,
                 delta_aggregation: float = 0.5,
                 delta_discrimination: float = 1.5) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.delta_aggregation = delta_aggregation
        self.delta_discrimination = delta_discrimination
        self.weights = (1.0, 1.0)

    def _forward_single(self, emb: torch.Tensor, instance: torch.Tensor,
                        kernel: torch.Tensor,
                        training_mask: torch.Tensor) -> torch.Tensor:
        """Compute the loss for a single image.

        Args:
            emb (torch.Tensor): The embedding feature.
            instance (torch.Tensor): The instance feature.
            kernel (torch.Tensor): The kernel feature.
            training_mask (torch.Tensor): The effective mask.
        """
        training_mask = (training_mask > 0.5).float()
        kernel = (kernel > 0.5).float()
        instance = instance * training_mask
        instance_kernel = (instance * kernel).view(-1)
        instance = instance.view(-1)
        emb = emb.view(self.feature_dim, -1)

        unique_labels, unique_ids = torch.unique(
            instance_kernel, sorted=True, return_inverse=True)
        num_instance = unique_labels.size(0)
        if num_instance <= 1:
            return 0

        emb_mean = emb.new_zeros((self.feature_dim, num_instance),
                                 dtype=torch.float32)
        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue
            ind_k = instance_kernel == lb
            emb_mean[:, i] = torch.mean(emb[:, ind_k], dim=1)

        l_agg = emb.new_zeros(num_instance, dtype=torch.float32)
        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue
            ind = instance == lb
            emb_ = emb[:, ind]
            dist = (emb_ - emb_mean[:, i:i + 1]).norm(p=2, dim=0)
            dist = F.relu(dist - self.delta_aggregation)**2
            l_agg[i] = torch.mean(torch.log(dist + 1.0))
        l_agg = torch.mean(l_agg[1:])

        if num_instance > 2:
            emb_interleave = emb_mean.permute(1, 0).repeat(num_instance, 1)
            emb_band = emb_mean.permute(1, 0).repeat(1, num_instance).view(
                -1, self.feature_dim)

            mask = (1 - torch.eye(num_instance, dtype=torch.int8)).view(
                -1, 1).repeat(1, self.feature_dim)
            mask = mask.view(num_instance, num_instance, -1)
            mask[0, :, :] = 0
            mask[:, 0, :] = 0
            mask = mask.view(num_instance * num_instance, -1)

            dist = emb_interleave - emb_band
            dist = dist[mask > 0].view(-1, self.feature_dim).norm(p=2, dim=1)
            dist = F.relu(2 * self.delta_discrimination - dist)**2
            l_dis = torch.mean(torch.log(dist + 1.0))
        else:
            l_dis = 0

        l_agg = self.weights[0] * l_agg
        l_dis = self.weights[1] * l_dis
        l_reg = torch.mean(torch.log(torch.norm(emb_mean, 2, 0) + 1.0)) * 0.001
        loss = l_agg + l_dis + l_reg
        return loss

    def forward(self, emb: torch.Tensor, instance: torch.Tensor,
                kernel: torch.Tensor,
                training_mask: torch.Tensor) -> torch.Tensor:
        """Compute the loss for a batch image.

        Args:
            emb (torch.Tensor): The embedding feature.
            instance (torch.Tensor): The instance feature.
            kernel (torch.Tensor): The kernel feature.
            training_mask (torch.Tensor): The effective mask.
        """
        loss_batch = emb.new_zeros((emb.size(0)), dtype=torch.float32)

        for i in range(loss_batch.size(0)):
            loss_batch[i] = self._forward_single(emb[i], instance[i],
                                                 kernel[i], training_mask[i])

        return loss_batch
