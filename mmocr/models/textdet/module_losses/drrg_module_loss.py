# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from lanms import merge_quadrangle_n9 as la_nms
from mmcv.image import imrescale
from mmdet.models.utils import multi_apply
from numpy import ndarray
from numpy.linalg import norm
from torch import Tensor

from mmocr.registry import MODELS
from mmocr.structures import TextDetDataSample
from mmocr.utils import check_argument
from .textsnake_module_loss import TextSnakeModuleLoss


@MODELS.register_module()
class DRRGModuleLoss(TextSnakeModuleLoss):
    """The class for implementing DRRG loss. This is partially adapted from
    https://github.com/GXYM/DRRG licensed under the MIT license.

    DRRG: `Deep Relational Reasoning Graph Network for Arbitrary Shape Text
    Detection <https://arxiv.org/abs/1908.05900>`_.

    Args:
        ohem_ratio (float): The negative/positive ratio in ohem. Defaults to
            3.0.
        downsample_ratio (float): Downsample ratio. Defaults to 1.0. TODO:
            remove it.
        orientation_thr (float): The threshold for distinguishing between
            head edge and tail edge among the horizontal and vertical edges
            of a quadrangle. Defaults to 2.0.
        resample_step (float): The step size for resampling the text center
            line. Defaults to 8.0.
        num_min_comps (int): The minimum number of text components, which
            should be larger than k_hop1 mentioned in paper. Defaults to 9.
        num_max_comps (int): The maximum number of text components. Defaults
            to 600.
        min_width (float): The minimum width of text components. Defaults to
            8.0.
        max_width (float): The maximum width of text components. Defaults to
            24.0.
        center_region_shrink_ratio (float): The shrink ratio of text center
            regions. Defaults to 0.3.
        comp_shrink_ratio (float): The shrink ratio of text components.
            Defaults to 1.0.
        comp_w_h_ratio (float): The width to height ratio of text components.
            Defaults to 0.3.
        min_rand_half_height(float): The minimum half-height of random text
            components. Defaults to 8.0.
        max_rand_half_height (float): The maximum half-height of random
            text components. Defaults to 24.0.
        jitter_level (float): The jitter level of text component geometric
            features. Defaults to 0.2.
        loss_text (dict): The loss config used to calculate the text loss.
            Defaults to ``dict(type='MaskedBalancedBCEWithLogitsLoss',
            fallback_negative_num=100, eps=1e-5)``.
        loss_center (dict): The loss config used to calculate the center loss.
            Defaults to ``dict(type='MaskedBCEWithLogitsLoss')``.
        loss_top (dict): The loss config used to calculate the top loss, which
            is a part of the height loss. Defaults to
            ``dict(type='SmoothL1Loss', reduction='none')``.
        loss_btm (dict): The loss config used to calculate the bottom loss,
            which is a part of the height loss. Defaults to
            ``dict(type='SmoothL1Loss', reduction='none')``.
        loss_sin (dict): The loss config used to calculate the sin loss.
            Defaults to ``dict(type='MaskedSmoothL1Loss')``.
        loss_cos (dict): The loss config used to calculate the cos loss.
            Defaults to ``dict(type='MaskedSmoothL1Loss')``.
        loss_gcn (dict): The loss config used to calculate the GCN loss.
            Defaults to ``dict(type='CrossEntropyLoss')``.
    """

    def __init__(
        self,
        ohem_ratio: float = 3.0,
        downsample_ratio: float = 1.0,
        orientation_thr: float = 2.0,
        resample_step: float = 8.0,
        num_min_comps: int = 9,
        num_max_comps: int = 600,
        min_width: float = 8.0,
        max_width: float = 24.0,
        center_region_shrink_ratio: float = 0.3,
        comp_shrink_ratio: float = 1.0,
        comp_w_h_ratio: float = 0.3,
        text_comp_nms_thr: float = 0.25,
        min_rand_half_height: float = 8.0,
        max_rand_half_height: float = 24.0,
        jitter_level: float = 0.2,
        loss_text: Dict = dict(
            type='MaskedBalancedBCEWithLogitsLoss',
            fallback_negative_num=100,
            eps=1e-5),
        loss_center: Dict = dict(type='MaskedBCEWithLogitsLoss'),
        loss_top: Dict = dict(type='SmoothL1Loss', reduction='none'),
        loss_btm: Dict = dict(type='SmoothL1Loss', reduction='none'),
        loss_sin: Dict = dict(type='MaskedSmoothL1Loss'),
        loss_cos: Dict = dict(type='MaskedSmoothL1Loss'),
        loss_gcn: Dict = dict(type='CrossEntropyLoss')
    ) -> None:
        super().__init__()
        self.ohem_ratio = ohem_ratio
        self.downsample_ratio = downsample_ratio
        self.orientation_thr = orientation_thr
        self.resample_step = resample_step
        self.num_max_comps = num_max_comps
        self.num_min_comps = num_min_comps
        self.min_width = min_width
        self.max_width = max_width
        self.center_region_shrink_ratio = center_region_shrink_ratio
        self.comp_shrink_ratio = comp_shrink_ratio
        self.comp_w_h_ratio = comp_w_h_ratio
        self.text_comp_nms_thr = text_comp_nms_thr
        self.min_rand_half_height = min_rand_half_height
        self.max_rand_half_height = max_rand_half_height
        self.jitter_level = jitter_level
        self.loss_text = MODELS.build(loss_text)
        self.loss_center = MODELS.build(loss_center)
        self.loss_top = MODELS.build(loss_top)
        self.loss_btm = MODELS.build(loss_btm)
        self.loss_sin = MODELS.build(loss_sin)
        self.loss_cos = MODELS.build(loss_cos)
        self.loss_gcn = MODELS.build(loss_gcn)

    def forward(self, preds: Tuple[Tensor, Tensor, Tensor],
                data_samples: Sequence[TextDetDataSample]) -> Dict:
        """Compute Drrg loss.

        Args:
            preds (tuple): The prediction
                tuple(pred_maps, gcn_pred, gt_labels), each of shape
                :math:`(N, 6, H, W)`, :math:`(N, 2)` and :math:`(m ,n)`, where
                :math:`m * n = N`.
            data_samples (list[TextDetDataSample]): The data samples.

        Returns:
            dict:  A loss dict with ``loss_text``, ``loss_center``,
            ``loss_height``, ``loss_sin``, ``loss_cos``, and ``loss_gcn``.
        """
        assert isinstance(preds, tuple)

        (gt_text_masks, gt_center_region_masks, gt_masks, gt_top_height_maps,
         gt_bot_height_maps, gt_sin_maps, gt_cos_maps,
         _) = self.get_targets(data_samples)
        pred_maps, gcn_pred, gt_labels = preds
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
            'gt_text_masks': gt_text_masks,
            'gt_center_region_masks': gt_center_region_masks,
            'gt_masks': gt_masks,
            'gt_top_height_maps': gt_top_height_maps,
            'gt_bot_height_maps': gt_bot_height_maps,
            'gt_sin_maps': gt_sin_maps,
            'gt_cos_maps': gt_cos_maps
        }
        gt = {}
        for key, value in mapping.items():
            gt[key] = value
            if abs(self.downsample_ratio - 1.0) < 1e-2:
                gt[key] = self._batch_pad(gt[key], feature_sz[2:])
            else:
                gt[key] = [
                    imrescale(
                        mask,
                        scale=self.downsample_ratio,
                        interpolation='nearest') for mask in gt[key]
                ]
                gt[key] = self._batch_pad(gt[key], feature_sz[2:])
                if key in ['gt_top_height_maps', 'gt_bot_height_maps']:
                    gt[key] *= self.downsample_ratio
            gt[key] = torch.from_numpy(gt[key]).float().to(device)

        scale = torch.sqrt(1.0 / (pred_sin_map**2 + pred_cos_map**2 + 1e-8))
        pred_sin_map = pred_sin_map * scale
        pred_cos_map = pred_cos_map * scale

        loss_text = self.loss_text(pred_text_region, gt['gt_text_masks'],
                                   gt['gt_masks'])

        text_mask = (gt['gt_text_masks'] * gt['gt_masks']).float()
        negative_text_mask = ((1 - gt['gt_text_masks']) *
                              gt['gt_masks']).float()
        loss_center_positive = self.loss_center(pred_center_region,
                                                gt['gt_center_region_masks'],
                                                text_mask)
        loss_center_negative = self.loss_center(pred_center_region,
                                                gt['gt_center_region_masks'],
                                                negative_text_mask)
        loss_center = loss_center_positive + 0.5 * loss_center_negative

        center_mask = (gt['gt_center_region_masks'] * gt['gt_masks']).float()
        map_sz = pred_top_height_map.size()
        ones = torch.ones(map_sz, dtype=torch.float, device=device)
        loss_top = self.loss_top(
            pred_top_height_map / (gt['gt_top_height_maps'] + 1e-2), ones)
        loss_btm = self.loss_btm(
            pred_bot_height_map / (gt['gt_bot_height_maps'] + 1e-2), ones)
        gt_height = gt['gt_top_height_maps'] + gt['gt_bot_height_maps']
        loss_height = torch.sum((torch.log(gt_height + 1) *
                                 (loss_top + loss_btm)) * center_mask) / (
                                     torch.sum(center_mask) + 1e-6)

        loss_sin = self.loss_sin(pred_sin_map, gt['gt_sin_maps'], center_mask)
        loss_cos = self.loss_cos(pred_cos_map, gt['gt_cos_maps'], center_mask)

        loss_gcn = self.loss_gcn(gcn_pred,
                                 gt_labels.view(-1).to(gcn_pred.device))

        results = dict(
            loss_text=loss_text,
            loss_center=loss_center,
            loss_height=loss_height,
            loss_sin=loss_sin,
            loss_cos=loss_cos,
            loss_gcn=loss_gcn)

        return results

    def get_targets(self, data_samples: List[TextDetDataSample]) -> Tuple:
        """Generate loss targets from data samples.

        Args:
            data_samples (list(TextDetDataSample)): Ground truth data samples.

        Returns:
            tuple: A tuple of 8 lists of tensors as DRRG targets. Read
            docstring of ``_get_target_single`` for more details.
        """

        # If data_samples points to same object as self.cached_data_samples, it
        # means that get_targets is called more than once in the same train
        # iteration, and pre-computed targets can be reused.
        if hasattr(self, 'targets') and \
                self.cache_data_samples is data_samples:
            return self.targets

        self.cache_data_samples = data_samples
        self.targets = multi_apply(self._get_target_single, data_samples)
        return self.targets

    def _get_target_single(self, data_sample: TextDetDataSample) -> Tuple:
        """Generate loss target from a data sample.

        Args:
            data_sample (TextDetDataSample): The data sample.

        Returns:
            tuple: A tuple of 8 tensors as DRRG targets.

            - gt_text_mask (ndarray): The text region mask.
            - gt_center_region_mask (ndarray): The text center region mask.
            - gt_mask (ndarray): The effective mask.
            - gt_top_height_map (ndarray): The map on which the distance from
              points to top side lines will be drawn for each pixel in text
              center regions.
            - gt_bot_height_map (ndarray): The map on which the distance from
              points to bottom side lines will be drawn for each pixel in text
              center regions.
            - gt_sin_map (ndarray): The sin(theta) map where theta is the angle
              between vector (top point - bottom point) and vector (1, 0).
            - gt_cos_map (ndarray): The cos(theta) map where theta is the angle
              between vector (top point - bottom point) and vector (1, 0).
            - gt_comp_attribs (ndarray): The padded text component attributes
              of a fixed size. Shape: (num_component, 8).
        """

        gt_instances = data_sample.gt_instances
        ignore_flags = gt_instances.ignored

        polygons = gt_instances[~ignore_flags].polygons
        ignored_polygons = gt_instances[ignore_flags].polygons
        h, w = data_sample.img_shape

        gt_text_mask = self._generate_text_region_mask((h, w), polygons)
        gt_mask = self._generate_effective_mask((h, w), ignored_polygons)
        (center_lines, gt_center_region_mask, gt_top_height_map,
         gt_bot_height_map, gt_sin_map,
         gt_cos_map) = self._generate_center_mask_attrib_maps((h, w), polygons)

        gt_comp_attribs = self._generate_comp_attribs(center_lines,
                                                      gt_text_mask,
                                                      gt_center_region_mask,
                                                      gt_top_height_map,
                                                      gt_bot_height_map,
                                                      gt_sin_map, gt_cos_map)

        return (gt_text_mask, gt_center_region_mask, gt_mask,
                gt_top_height_map, gt_bot_height_map, gt_sin_map, gt_cos_map,
                gt_comp_attribs)

    def _generate_center_mask_attrib_maps(self, img_size: Tuple[int, int],
                                          text_polys: List[ndarray]) -> Tuple:
        """Generate text center region masks and geometric attribute maps.

        Args:
            img_size (tuple(int, int)): The image size (height, width).
            text_polys (list[ndarray]): The list of text polygons.

        Returns:
            tuple(center_lines, center_region_mask, top_height_map,
            bot_height_map,sin_map, cos_map):

            center_lines (list[ndarray]): The list of text center lines.
            center_region_mask (ndarray): The text center region mask.
            top_height_map (ndarray): The map on which the distance from points
                to top side lines will be drawn for each pixel in text center
                regions.
            bot_height_map (ndarray): The map on which the distance from points
                to bottom side lines will be drawn for each pixel in text
                center regions.
            sin_map (ndarray): The sin(theta) map where theta is the angle
                between vector (top point - bottom point) and vector (1, 0).
            cos_map (ndarray): The cos(theta) map where theta is the angle
                between vector (top point - bottom point) and vector (1, 0).
        """

        assert isinstance(img_size, tuple)
        assert check_argument.is_type_list(text_polys, ndarray)

        h, w = img_size

        center_lines = []
        center_region_mask = np.zeros((h, w), np.uint8)
        top_height_map = np.zeros((h, w), dtype=np.float32)
        bot_height_map = np.zeros((h, w), dtype=np.float32)
        sin_map = np.zeros((h, w), dtype=np.float32)
        cos_map = np.zeros((h, w), dtype=np.float32)

        for poly in text_polys:
            polygon_points = poly.reshape(-1, 2)
            _, _, top_line, bot_line = self._reorder_poly_edge(polygon_points)
            resampled_top_line, resampled_bot_line = self._resample_sidelines(
                top_line, bot_line, self.resample_step)
            resampled_bot_line = resampled_bot_line[::-1]
            center_line = (resampled_top_line + resampled_bot_line) / 2

            if self.vector_slope(center_line[-1] - center_line[0]) > 2:
                if (center_line[-1] - center_line[0])[1] < 0:
                    center_line = center_line[::-1]
                    resampled_top_line = resampled_top_line[::-1]
                    resampled_bot_line = resampled_bot_line[::-1]
            else:
                if (center_line[-1] - center_line[0])[0] < 0:
                    center_line = center_line[::-1]
                    resampled_top_line = resampled_top_line[::-1]
                    resampled_bot_line = resampled_bot_line[::-1]

            line_head_shrink_len = np.clip(
                (norm(top_line[0] - bot_line[0]) * self.comp_w_h_ratio),
                self.min_width, self.max_width) / 2
            line_tail_shrink_len = np.clip(
                (norm(top_line[-1] - bot_line[-1]) * self.comp_w_h_ratio),
                self.min_width, self.max_width) / 2
            num_head_shrink = int(line_head_shrink_len // self.resample_step)
            num_tail_shrink = int(line_tail_shrink_len // self.resample_step)
            if len(center_line) > num_head_shrink + num_tail_shrink + 2:
                center_line = center_line[num_head_shrink:len(center_line) -
                                          num_tail_shrink]
                resampled_top_line = resampled_top_line[
                    num_head_shrink:len(resampled_top_line) - num_tail_shrink]
                resampled_bot_line = resampled_bot_line[
                    num_head_shrink:len(resampled_bot_line) - num_tail_shrink]
            center_lines.append(center_line.astype(np.int32))

            self._draw_center_region_maps(resampled_top_line,
                                          resampled_bot_line, center_line,
                                          center_region_mask, top_height_map,
                                          bot_height_map, sin_map, cos_map,
                                          self.center_region_shrink_ratio)

        return (center_lines, center_region_mask, top_height_map,
                bot_height_map, sin_map, cos_map)

    def _generate_comp_attribs(self, center_lines: List[ndarray],
                               text_mask: ndarray, center_region_mask: ndarray,
                               top_height_map: ndarray,
                               bot_height_map: ndarray, sin_map: ndarray,
                               cos_map: ndarray) -> ndarray:
        """Generate text component attributes.

        Args:
            center_lines (list[ndarray]): The list of text center lines .
            text_mask (ndarray): The text region mask.
            center_region_mask (ndarray): The text center region mask.
            top_height_map (ndarray): The map on which the distance from points
                to top side lines will be drawn for each pixel in text center
                regions.
            bot_height_map (ndarray): The map on which the distance from points
                to bottom side lines will be drawn for each pixel in text
                center regions.
            sin_map (ndarray): The sin(theta) map where theta is the angle
                between vector (top point - bottom point) and vector (1, 0).
            cos_map (ndarray): The cos(theta) map where theta is the angle
                between vector (top point - bottom point) and vector (1, 0).

        Returns:
            ndarray: The padded text component attributes of a fixed size.
        """

        assert isinstance(center_lines, list)
        assert (text_mask.shape == center_region_mask.shape ==
                top_height_map.shape == bot_height_map.shape == sin_map.shape
                == cos_map.shape)

        center_lines_mask = np.zeros_like(center_region_mask)
        cv2.polylines(center_lines_mask, center_lines, 0, 1, 1)
        center_lines_mask = center_lines_mask * center_region_mask
        comp_centers = np.argwhere(center_lines_mask > 0)

        y = comp_centers[:, 0]
        x = comp_centers[:, 1]

        top_height = top_height_map[y, x].reshape(
            (-1, 1)) * self.comp_shrink_ratio
        bot_height = bot_height_map[y, x].reshape(
            (-1, 1)) * self.comp_shrink_ratio
        sin = sin_map[y, x].reshape((-1, 1))
        cos = cos_map[y, x].reshape((-1, 1))

        top_mid_points = comp_centers + np.hstack(
            [top_height * sin, top_height * cos])
        bot_mid_points = comp_centers - np.hstack(
            [bot_height * sin, bot_height * cos])

        width = (top_height + bot_height) * self.comp_w_h_ratio
        width = np.clip(width, self.min_width, self.max_width)
        r = width / 2

        tl = top_mid_points[:, ::-1] - np.hstack([-r * sin, r * cos])
        tr = top_mid_points[:, ::-1] + np.hstack([-r * sin, r * cos])
        br = bot_mid_points[:, ::-1] + np.hstack([-r * sin, r * cos])
        bl = bot_mid_points[:, ::-1] - np.hstack([-r * sin, r * cos])
        text_comps = np.hstack([tl, tr, br, bl]).astype(np.float32)

        score = np.ones((text_comps.shape[0], 1), dtype=np.float32)
        text_comps = np.hstack([text_comps, score])
        text_comps = la_nms(text_comps, self.text_comp_nms_thr)

        if text_comps.shape[0] >= 1:
            img_h, img_w = center_region_mask.shape
            text_comps[:, 0:8:2] = np.clip(text_comps[:, 0:8:2], 0, img_w - 1)
            text_comps[:, 1:8:2] = np.clip(text_comps[:, 1:8:2], 0, img_h - 1)

            comp_centers = np.mean(
                text_comps[:, 0:8].reshape((-1, 4, 2)),
                axis=1).astype(np.int32)
            x = comp_centers[:, 0]
            y = comp_centers[:, 1]

            height = (top_height_map[y, x] + bot_height_map[y, x]).reshape(
                (-1, 1))
            width = np.clip(height * self.comp_w_h_ratio, self.min_width,
                            self.max_width)

            cos = cos_map[y, x].reshape((-1, 1))
            sin = sin_map[y, x].reshape((-1, 1))

            _, comp_label_mask = cv2.connectedComponents(
                center_region_mask, connectivity=8)
            comp_labels = comp_label_mask[y, x].reshape(
                (-1, 1)).astype(np.float32)

            x = x.reshape((-1, 1)).astype(np.float32)
            y = y.reshape((-1, 1)).astype(np.float32)
            comp_attribs = np.hstack(
                [x, y, height, width, cos, sin, comp_labels])
            comp_attribs = self._jitter_comp_attribs(comp_attribs,
                                                     self.jitter_level)

            if comp_attribs.shape[0] < self.num_min_comps:
                num_rand_comps = self.num_min_comps - comp_attribs.shape[0]
                rand_comp_attribs = self._generate_rand_comp_attribs(
                    num_rand_comps, 1 - text_mask)
                comp_attribs = np.vstack([comp_attribs, rand_comp_attribs])
        else:
            comp_attribs = self._generate_rand_comp_attribs(
                self.num_min_comps, 1 - text_mask)

        num_comps = (
            np.ones((comp_attribs.shape[0], 1), dtype=np.float32) *
            comp_attribs.shape[0])
        comp_attribs = np.hstack([num_comps, comp_attribs])

        if comp_attribs.shape[0] > self.num_max_comps:
            comp_attribs = comp_attribs[:self.num_max_comps, :]
            comp_attribs[:, 0] = self.num_max_comps

        pad_comp_attribs = np.zeros(
            (self.num_max_comps, comp_attribs.shape[1]), dtype=np.float32)
        pad_comp_attribs[:comp_attribs.shape[0], :] = comp_attribs

        return pad_comp_attribs

    def _generate_rand_comp_attribs(self, num_rand_comps: int,
                                    center_sample_mask: ndarray) -> ndarray:
        """Generate random text components and their attributes to ensure the
        the number of text components in an image is larger than k_hop1, which
        is the number of one hop neighbors in KNN graph.

        Args:
            num_rand_comps (int): The number of random text components.
            center_sample_mask (ndarray): The region mask for sampling text
                component centers .

        Returns:
            ndarray: The random text component attributes
            (x, y, h, w, cos, sin, comp_label=0).
        """

        assert isinstance(num_rand_comps, int)
        assert num_rand_comps > 0
        assert center_sample_mask.ndim == 2

        h, w = center_sample_mask.shape

        max_rand_half_height = self.max_rand_half_height
        min_rand_half_height = self.min_rand_half_height
        max_rand_height = max_rand_half_height * 2
        max_rand_width = np.clip(max_rand_height * self.comp_w_h_ratio,
                                 self.min_width, self.max_width)
        margin = int(
            np.sqrt((max_rand_height / 2)**2 + (max_rand_width / 2)**2)) + 1

        if 2 * margin + 1 > min(h, w):

            assert min(h, w) > (np.sqrt(2) * (self.min_width + 1))
            max_rand_half_height = max(min(h, w) / 4, self.min_width / 2 + 1)
            min_rand_half_height = max(max_rand_half_height / 4,
                                       self.min_width / 2)

            max_rand_height = max_rand_half_height * 2
            max_rand_width = np.clip(max_rand_height * self.comp_w_h_ratio,
                                     self.min_width, self.max_width)
            margin = int(
                np.sqrt((max_rand_height / 2)**2 +
                        (max_rand_width / 2)**2)) + 1

        inner_center_sample_mask = np.zeros_like(center_sample_mask)
        inner_center_sample_mask[margin:h - margin, margin:w - margin] = \
            center_sample_mask[margin:h - margin, margin:w - margin]
        kernel_size = int(np.clip(max_rand_half_height, 7, 21))
        inner_center_sample_mask = cv2.erode(
            inner_center_sample_mask,
            np.ones((kernel_size, kernel_size), np.uint8))

        center_candidates = np.argwhere(inner_center_sample_mask > 0)
        num_center_candidates = len(center_candidates)
        sample_inds = np.random.choice(num_center_candidates, num_rand_comps)
        rand_centers = center_candidates[sample_inds]

        rand_top_height = np.random.randint(
            min_rand_half_height,
            max_rand_half_height,
            size=(len(rand_centers), 1))
        rand_bot_height = np.random.randint(
            min_rand_half_height,
            max_rand_half_height,
            size=(len(rand_centers), 1))

        rand_cos = 2 * np.random.random(size=(len(rand_centers), 1)) - 1
        rand_sin = 2 * np.random.random(size=(len(rand_centers), 1)) - 1
        scale = np.sqrt(1.0 / (rand_cos**2 + rand_sin**2 + 1e-8))
        rand_cos = rand_cos * scale
        rand_sin = rand_sin * scale

        height = (rand_top_height + rand_bot_height)
        width = np.clip(height * self.comp_w_h_ratio, self.min_width,
                        self.max_width)

        rand_comp_attribs = np.hstack([
            rand_centers[:, ::-1], height, width, rand_cos, rand_sin,
            np.zeros_like(rand_sin)
        ]).astype(np.float32)

        return rand_comp_attribs

    def _jitter_comp_attribs(self, comp_attribs: ndarray,
                             jitter_level: float) -> ndarray:
        """Jitter text components attributes.

        Args:
            comp_attribs (ndarray): The text component attributes.
            jitter_level (float): The jitter level of text components
                attributes.

        Returns:
            ndarray: The jittered text component
            attributes (x, y, h, w, cos, sin, comp_label).
        """

        assert comp_attribs.shape[1] == 7
        assert comp_attribs.shape[0] > 0
        assert isinstance(jitter_level, float)

        x = comp_attribs[:, 0].reshape((-1, 1))
        y = comp_attribs[:, 1].reshape((-1, 1))
        h = comp_attribs[:, 2].reshape((-1, 1))
        w = comp_attribs[:, 3].reshape((-1, 1))
        cos = comp_attribs[:, 4].reshape((-1, 1))
        sin = comp_attribs[:, 5].reshape((-1, 1))
        comp_labels = comp_attribs[:, 6].reshape((-1, 1))

        x += (np.random.random(size=(len(comp_attribs), 1)) -
              0.5) * (h * np.abs(cos) + w * np.abs(sin)) * jitter_level
        y += (np.random.random(size=(len(comp_attribs), 1)) -
              0.5) * (h * np.abs(sin) + w * np.abs(cos)) * jitter_level

        h += (np.random.random(size=(len(comp_attribs), 1)) -
              0.5) * h * jitter_level
        w += (np.random.random(size=(len(comp_attribs), 1)) -
              0.5) * w * jitter_level

        cos += (np.random.random(size=(len(comp_attribs), 1)) -
                0.5) * 2 * jitter_level
        sin += (np.random.random(size=(len(comp_attribs), 1)) -
                0.5) * 2 * jitter_level

        scale = np.sqrt(1.0 / (cos**2 + sin**2 + 1e-8))
        cos = cos * scale
        sin = sin * scale

        jittered_comp_attribs = np.hstack([x, y, h, w, cos, sin, comp_labels])

        return jittered_comp_attribs

    def _draw_center_region_maps(self, top_line: ndarray, bot_line: ndarray,
                                 center_line: ndarray,
                                 center_region_mask: ndarray,
                                 top_height_map: ndarray,
                                 bot_height_map: ndarray, sin_map: ndarray,
                                 cos_map: ndarray,
                                 region_shrink_ratio: float) -> None:
        """Draw attributes of text components on text center regions.

        Args:
            top_line (ndarray): The points composing the top side lines of text
                polygons.
            bot_line (ndarray): The points composing bottom side lines of text
                polygons.
            center_line (ndarray): The points composing the center lines of
                text instances.
            center_region_mask (ndarray): The text center region mask.
            top_height_map (ndarray): The map on which the distance from points
                to top side lines will be drawn for each pixel in text center
                regions.
            bot_height_map (ndarray): The map on which the distance from points
                to bottom side lines will be drawn for each pixel in text
                center regions.
            sin_map (ndarray): The map of vector_sin(top_point - bot_point)
                that will be drawn on text center regions.
            cos_map (ndarray): The map of vector_cos(top_point - bot_point)
                will be drawn on text center regions.
            region_shrink_ratio (float): The shrink ratio of text center
                regions.
        """

        assert top_line.shape == bot_line.shape == center_line.shape
        assert (center_region_mask.shape == top_height_map.shape ==
                bot_height_map.shape == sin_map.shape == cos_map.shape)
        assert isinstance(region_shrink_ratio, float)

        h, w = center_region_mask.shape
        for i in range(0, len(center_line) - 1):

            top_mid_point = (top_line[i] + top_line[i + 1]) / 2
            bot_mid_point = (bot_line[i] + bot_line[i + 1]) / 2

            sin_theta = self.vector_sin(top_mid_point - bot_mid_point)
            cos_theta = self.vector_cos(top_mid_point - bot_mid_point)

            tl = center_line[i] + (top_line[i] -
                                   center_line[i]) * region_shrink_ratio
            tr = center_line[i + 1] + (
                top_line[i + 1] - center_line[i + 1]) * region_shrink_ratio
            br = center_line[i + 1] + (
                bot_line[i + 1] - center_line[i + 1]) * region_shrink_ratio
            bl = center_line[i] + (bot_line[i] -
                                   center_line[i]) * region_shrink_ratio
            current_center_box = np.vstack([tl, tr, br, bl]).astype(np.int32)

            cv2.fillPoly(center_region_mask, [current_center_box], color=1)
            cv2.fillPoly(sin_map, [current_center_box], color=sin_theta)
            cv2.fillPoly(cos_map, [current_center_box], color=cos_theta)

            current_center_box[:, 0] = np.clip(current_center_box[:, 0], 0,
                                               w - 1)
            current_center_box[:, 1] = np.clip(current_center_box[:, 1], 0,
                                               h - 1)
            min_coord = np.min(current_center_box, axis=0).astype(np.int32)
            max_coord = np.max(current_center_box, axis=0).astype(np.int32)
            current_center_box = current_center_box - min_coord
            box_sz = (max_coord - min_coord + 1)

            center_box_mask = np.zeros((box_sz[1], box_sz[0]), dtype=np.uint8)
            cv2.fillPoly(center_box_mask, [current_center_box], color=1)

            inds = np.argwhere(center_box_mask > 0)
            inds = inds + (min_coord[1], min_coord[0])
            inds_xy = np.fliplr(inds)
            top_height_map[(inds[:, 0], inds[:, 1])] = self._dist_point2line(
                inds_xy, (top_line[i], top_line[i + 1]))
            bot_height_map[(inds[:, 0], inds[:, 1])] = self._dist_point2line(
                inds_xy, (bot_line[i], bot_line[i + 1]))

    def _dist_point2line(self, point: ndarray,
                         line: Tuple[ndarray, ndarray]) -> ndarray:
        """Calculate the distance from points to a line.

        TODO: Check its mergibility with the one in mmocr.utils.point_utils.
        """

        assert isinstance(line, tuple)
        point1, point2 = line
        d = abs(np.cross(point2 - point1, point - point1)) / (
            norm(point2 - point1) + 1e-8)
        return d
