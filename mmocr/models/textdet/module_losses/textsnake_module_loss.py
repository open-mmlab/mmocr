# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from mmcv.image import impad, imrescale
from mmdet.models.utils import multi_apply
from numpy import ndarray
from numpy.linalg import norm
from torch import Tensor, nn

from mmocr.registry import MODELS
from mmocr.structures import TextDetDataSample
from .text_kernel_mixin import TextKernelMixin


@MODELS.register_module()
class TextSnakeModuleLoss(nn.Module, TextKernelMixin):
    """The class for implementing TextSnake loss. This is partially adapted
    from https://github.com/princewang1994/TextSnake.pytorch.

    TextSnake: `A Flexible Representation for Detecting Text of Arbitrary
    Shapes <https://arxiv.org/abs/1807.01544>`_.

    Args:
        ohem_ratio (float): The negative/positive ratio in ohem.
        downsample_ratio (float): Downsample ratio. Defaults to 1.0. TODO:
            remove it.
        orientation_thr (float): The threshold for distinguishing between
            head edge and tail edge among the horizontal and vertical edges
            of a quadrangle.
        resample_step (float): The step of resampling.
        center_region_shrink_ratio (float): The shrink ratio of text center.
        loss_text (dict): The loss config used to calculate the text loss.
        loss_center (dict): The loss config used to calculate the center loss.
        loss_radius (dict): The loss config used to calculate the radius loss.
        loss_sin (dict): The loss config used to calculate the sin loss.
        loss_cos (dict): The loss config used to calculate the cos loss.
    """

    def __init__(
        self,
        ohem_ratio: float = 3.0,
        downsample_ratio: float = 1.0,
        orientation_thr: float = 2.0,
        resample_step: float = 4.0,
        center_region_shrink_ratio: float = 0.3,
        loss_text: Dict = dict(
            type='MaskedBalancedBCEWithLogitsLoss',
            fallback_negative_num=100,
            eps=1e-5),
        loss_center: Dict = dict(type='MaskedBCEWithLogitsLoss'),
        loss_radius: Dict = dict(type='MaskedSmoothL1Loss'),
        loss_sin: Dict = dict(type='MaskedSmoothL1Loss'),
        loss_cos: Dict = dict(type='MaskedSmoothL1Loss')
    ) -> None:
        super().__init__()
        self.ohem_ratio = ohem_ratio
        self.downsample_ratio = downsample_ratio
        self.orientation_thr = orientation_thr
        self.resample_step = resample_step
        self.center_region_shrink_ratio = center_region_shrink_ratio
        self.eps = 1e-8
        self.loss_text = MODELS.build(loss_text)
        self.loss_center = MODELS.build(loss_center)
        self.loss_radius = MODELS.build(loss_radius)
        self.loss_sin = MODELS.build(loss_sin)
        self.loss_cos = MODELS.build(loss_cos)

    def _batch_pad(self, masks: List[ndarray],
                   target_sz: Tuple[int, int]) -> ndarray:
        """Pad the masks to the right and bottom side to the target size and
        pack them into a batch.

        Args:
            mask (list[ndarray]): The masks to be padded.
            target_sz (tuple(int, int)): The target tensor of size
                :math:`(H, W)`.

        Returns:
            ndarray: A batch of padded mask.
        """
        batch = []
        for mask in masks:
            # H x W
            mask_sz = mask.shape
            # left, top, right, bottom
            padding = (0, 0, target_sz[1] - mask_sz[1],
                       target_sz[0] - mask_sz[0])
            padded_mask = impad(
                mask, padding=padding, padding_mode='constant', pad_val=0)
            batch.append(np.expand_dims(padded_mask, axis=0))
        return np.concatenate(batch)

    def forward(self, preds: Tensor,
                data_samples: Sequence[TextDetDataSample]) -> Dict:
        """
        Args:
            preds (Tensor): The prediction map of shape
                :math:`(N, 5, H, W)`, where each dimension is the map of
                "text_region", "center_region", "sin_map", "cos_map", and
                "radius_map" respectively.
            data_samples (list[TextDetDataSample]): The data samples.

        Returns:
            dict:  A loss dict with ``loss_text``, ``loss_center``,
            ``loss_radius``, ``loss_sin`` and ``loss_cos``.
        """

        (gt_text_masks, gt_masks, gt_center_region_masks, gt_radius_maps,
         gt_sin_maps, gt_cos_maps) = self.get_targets(data_samples)

        pred_text_region = preds[:, 0, :, :]
        pred_center_region = preds[:, 1, :, :]
        pred_sin_map = preds[:, 2, :, :]
        pred_cos_map = preds[:, 3, :, :]
        pred_radius_map = preds[:, 4, :, :]
        feature_sz = preds.size()
        device = preds.device

        mapping = {
            'gt_text_masks': gt_text_masks,
            'gt_center_region_masks': gt_center_region_masks,
            'gt_masks': gt_masks,
            'gt_radius_maps': gt_radius_maps,
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
                if key == 'gt_radius_maps':
                    gt[key] *= self.downsample_ratio
            gt[key] = torch.from_numpy(gt[key]).float().to(device)

        scale = torch.sqrt(1.0 / (pred_sin_map**2 + pred_cos_map**2 + 1e-8))
        pred_sin_map = pred_sin_map * scale
        pred_cos_map = pred_cos_map * scale

        loss_text = self.loss_text(pred_text_region, gt['gt_text_masks'],
                                   gt['gt_masks'])

        text_mask = (gt['gt_text_masks'] * gt['gt_masks']).float()
        loss_center = self.loss_center(pred_center_region,
                                       gt['gt_center_region_masks'], text_mask)

        center_mask = (gt['gt_center_region_masks'] * gt['gt_masks']).float()
        map_sz = pred_radius_map.size()
        ones = torch.ones(map_sz, dtype=torch.float, device=device)
        loss_radius = self.loss_radius(
            pred_radius_map / (gt['gt_radius_maps'] + 1e-2), ones, center_mask)
        loss_sin = self.loss_sin(pred_sin_map, gt['gt_sin_maps'], center_mask)
        loss_cos = self.loss_cos(pred_cos_map, gt['gt_cos_maps'], center_mask)

        results = dict(
            loss_text=loss_text,
            loss_center=loss_center,
            loss_radius=loss_radius,
            loss_sin=loss_sin,
            loss_cos=loss_cos)

        return results

    def get_targets(self, data_samples: List[TextDetDataSample]) -> Tuple:
        """Generate loss targets from data samples.

        Args:
            data_samples (list(TextDetDataSample)): Ground truth data samples.

        Returns:
            tuple(gt_text_masks, gt_masks, gt_center_region_masks,
            gt_radius_maps, gt_sin_maps, gt_cos_maps):
            A tuple of six lists of ndarrays as the targets.
        """
        return multi_apply(self._get_target_single, data_samples)

    def _get_target_single(self, data_sample: TextDetDataSample) -> Tuple:
        """Generate loss target from a data sample.

        Args:
            data_sample (TextDetDataSample): The data sample.

        Returns:
            tuple(gt_text_mask, gt_mask, gt_center_region_mask, gt_radius_map,
            gt_sin_map, gt_cos_map):
            A tuple of six ndarrays as the targets of one prediction.
        """

        gt_instances = data_sample.gt_instances
        ignore_flags = gt_instances.ignored

        polygons = gt_instances[~ignore_flags].polygons
        ignored_polygons = gt_instances[ignore_flags].polygons

        gt_text_mask = self._generate_text_region_mask(data_sample.img_shape,
                                                       polygons)
        gt_mask = self._generate_effective_mask(data_sample.img_shape,
                                                ignored_polygons)

        (gt_center_region_mask, gt_radius_map, gt_sin_map,
         gt_cos_map) = self._generate_center_mask_attrib_maps(
             data_sample.img_shape, polygons)

        return (gt_text_mask, gt_mask, gt_center_region_mask, gt_radius_map,
                gt_sin_map, gt_cos_map)

    def _generate_text_region_mask(self, img_size: Tuple[int, int],
                                   text_polys: List[ndarray]) -> ndarray:
        """Generate text center region mask and geometry attribute maps.

        Args:
            img_size (tuple): The image size (height, width).
            text_polys (list[ndarray]): The list of text polygons.

        Returns:
            text_region_mask (ndarray): The text region mask.
        """

        assert isinstance(img_size, tuple)

        text_region_mask = np.zeros(img_size, dtype=np.uint8)

        for poly in text_polys:
            polygon = np.array(poly, dtype=np.int32).reshape((1, -1, 2))
            cv2.fillPoly(text_region_mask, polygon, 1)

        return text_region_mask

    def _generate_center_mask_attrib_maps(
        self, img_size: Tuple[int, int], text_polys: List[ndarray]
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """Generate text center region mask and geometric attribute maps.

        Args:
            img_size (tuple(int, int)): The image size of (height, width).
            text_polys (list[ndarray]): The list of text polygons.

        Returns:
            Tuple(center_region_mask, radius_map, sin_map, cos_map):

            - center_region_mask (ndarray): The text center region mask.
            - radius_map (ndarray): The distance map from each pixel in text
              center region to top sideline.
            - sin_map (ndarray): The sin(theta) map where theta is the angle
              between vector (top point - bottom point) and vector (1, 0).
            - cos_map (ndarray): The cos(theta) map where theta is the angle
              between vector (top point - bottom point) and vector (1, 0).
        """

        assert isinstance(img_size, tuple)

        center_region_mask = np.zeros(img_size, np.uint8)
        radius_map = np.zeros(img_size, dtype=np.float32)
        sin_map = np.zeros(img_size, dtype=np.float32)
        cos_map = np.zeros(img_size, dtype=np.float32)

        for poly in text_polys:
            polygon_points = np.array(poly).reshape(-1, 2)

            n = len(polygon_points)
            keep_inds = []
            for i in range(n):
                if norm(polygon_points[i] -
                        polygon_points[(i + 1) % n]) > 1e-5:
                    keep_inds.append(i)
            polygon_points = polygon_points[keep_inds]

            _, _, top_line, bot_line = self._reorder_poly_edge(polygon_points)
            resampled_top_line, resampled_bot_line = self._resample_sidelines(
                top_line, bot_line, self.resample_step)
            resampled_bot_line = resampled_bot_line[::-1]
            center_line = (resampled_top_line + resampled_bot_line) / 2

            if self.vector_slope(center_line[-1] - center_line[0]) > 0.9:
                if (center_line[-1] - center_line[0])[1] < 0:
                    center_line = center_line[::-1]
                    resampled_top_line = resampled_top_line[::-1]
                    resampled_bot_line = resampled_bot_line[::-1]
            else:
                if (center_line[-1] - center_line[0])[0] < 0:
                    center_line = center_line[::-1]
                    resampled_top_line = resampled_top_line[::-1]
                    resampled_bot_line = resampled_bot_line[::-1]

            line_head_shrink_len = norm(resampled_top_line[0] -
                                        resampled_bot_line[0]) / 4.0
            line_tail_shrink_len = norm(resampled_top_line[-1] -
                                        resampled_bot_line[-1]) / 4.0
            head_shrink_num = int(line_head_shrink_len // self.resample_step)
            tail_shrink_num = int(line_tail_shrink_len // self.resample_step)

            if len(center_line) > head_shrink_num + tail_shrink_num + 2:
                center_line = center_line[head_shrink_num:len(center_line) -
                                          tail_shrink_num]
                resampled_top_line = resampled_top_line[
                    head_shrink_num:len(resampled_top_line) - tail_shrink_num]
                resampled_bot_line = resampled_bot_line[
                    head_shrink_num:len(resampled_bot_line) - tail_shrink_num]

            self._draw_center_region_maps(resampled_top_line,
                                          resampled_bot_line, center_line,
                                          center_region_mask, radius_map,
                                          sin_map, cos_map,
                                          self.center_region_shrink_ratio)

        return center_region_mask, radius_map, sin_map, cos_map

    def _reorder_poly_edge(self, points: ndarray
                           ) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """Get the respective points composing head edge, tail edge, top
        sideline and bottom sideline.

        Args:
            points (ndarray): The points composing a text polygon.

        Returns:
            Tuple(center_region_mask, radius_map, sin_map, cos_map):

            - head_edge (ndarray): The two points composing the head edge of
              text polygon.
            - tail_edge (ndarray): The two points composing the tail edge of
              text polygon.
            - top_sideline (ndarray): The points composing top curved sideline
              of text polygon.
            - bot_sideline (ndarray): The points composing bottom curved
              sideline of text polygon.
        """

        assert points.ndim == 2
        assert points.shape[0] >= 4
        assert points.shape[1] == 2

        head_inds, tail_inds = self._find_head_tail(points,
                                                    self.orientation_thr)
        head_edge, tail_edge = points[head_inds], points[tail_inds]

        pad_points = np.vstack([points, points])
        if tail_inds[1] < 1:
            tail_inds[1] = len(points)
        sideline1 = pad_points[head_inds[1]:tail_inds[1]]
        sideline2 = pad_points[tail_inds[1]:(head_inds[1] + len(points))]
        sideline_mean_shift = np.mean(
            sideline1, axis=0) - np.mean(
                sideline2, axis=0)

        if sideline_mean_shift[1] > 0:
            top_sideline, bot_sideline = sideline2, sideline1
        else:
            top_sideline, bot_sideline = sideline1, sideline2

        return head_edge, tail_edge, top_sideline, bot_sideline

    def _find_head_tail(self, points: ndarray,
                        orientation_thr: float) -> Tuple[List[int], List[int]]:
        """Find the head edge and tail edge of a text polygon.

        Args:
            points (ndarray): The points composing a text polygon.
            orientation_thr (float): The threshold for distinguishing between
                head edge and tail edge among the horizontal and vertical edges
                of a quadrangle.

        Returns:
            Tuple(head_inds, tail_inds):

            - head_inds (list[int]): The indexes of two points composing head
              edge.
            - tail_inds (list[int]): The indexes of two points composing tail
              edge.
        """

        assert points.ndim == 2
        assert points.shape[0] >= 4
        assert points.shape[1] == 2
        assert isinstance(orientation_thr, float)

        if len(points) > 4:
            pad_points = np.vstack([points, points[0]])
            edge_vec = pad_points[1:] - pad_points[:-1]

            theta_sum = []
            adjacent_vec_theta = []
            for i, edge_vec1 in enumerate(edge_vec):
                adjacent_ind = [x % len(edge_vec) for x in [i - 1, i + 1]]
                adjacent_edge_vec = edge_vec[adjacent_ind]
                temp_theta_sum = np.sum(
                    self.vector_angle(edge_vec1, adjacent_edge_vec))
                temp_adjacent_theta = self.vector_angle(
                    adjacent_edge_vec[0], adjacent_edge_vec[1])
                theta_sum.append(temp_theta_sum)
                adjacent_vec_theta.append(temp_adjacent_theta)
            theta_sum_score = np.array(theta_sum) / np.pi
            adjacent_theta_score = np.array(adjacent_vec_theta) / np.pi
            poly_center = np.mean(points, axis=0)
            edge_dist = np.maximum(
                norm(pad_points[1:] - poly_center, axis=-1),
                norm(pad_points[:-1] - poly_center, axis=-1))
            dist_score = edge_dist / (np.max(edge_dist) + self.eps)
            position_score = np.zeros(len(edge_vec))
            score = 0.5 * theta_sum_score + 0.15 * adjacent_theta_score
            score += 0.35 * dist_score
            if len(points) % 2 == 0:
                position_score[(len(score) // 2 - 1)] += 1
                position_score[-1] += 1
            score += 0.1 * position_score
            pad_score = np.concatenate([score, score])
            score_matrix = np.zeros((len(score), len(score) - 3))
            x = np.arange(len(score) - 3) / float(len(score) - 4)
            gaussian = 1. / (np.sqrt(2. * np.pi) * 0.5) * np.exp(-np.power(
                (x - 0.5) / 0.5, 2.) / 2)
            gaussian = gaussian / np.max(gaussian)
            for i in range(len(score)):
                score_matrix[i, :] = score[i] + pad_score[
                    (i + 2):(i + len(score) - 1)] * gaussian * 0.3

            head_start, tail_increment = np.unravel_index(
                score_matrix.argmax(), score_matrix.shape)
            tail_start = (head_start + tail_increment + 2) % len(points)
            head_end = (head_start + 1) % len(points)
            tail_end = (tail_start + 1) % len(points)

            if head_end > tail_end:
                head_start, tail_start = tail_start, head_start
                head_end, tail_end = tail_end, head_end
            head_inds = [head_start, head_end]
            tail_inds = [tail_start, tail_end]
        else:
            if self.vector_slope(points[1] - points[0]) + self.vector_slope(
                    points[3] - points[2]) < self.vector_slope(
                        points[2] - points[1]) + self.vector_slope(points[0] -
                                                                   points[3]):
                horizontal_edge_inds = [[0, 1], [2, 3]]
                vertical_edge_inds = [[3, 0], [1, 2]]
            else:
                horizontal_edge_inds = [[3, 0], [1, 2]]
                vertical_edge_inds = [[0, 1], [2, 3]]

            vertical_len_sum = norm(points[vertical_edge_inds[0][0]] -
                                    points[vertical_edge_inds[0][1]]) + norm(
                                        points[vertical_edge_inds[1][0]] -
                                        points[vertical_edge_inds[1][1]])
            horizontal_len_sum = norm(
                points[horizontal_edge_inds[0][0]] -
                points[horizontal_edge_inds[0][1]]) + norm(
                    points[horizontal_edge_inds[1][0]] -
                    points[horizontal_edge_inds[1][1]])

            if vertical_len_sum > horizontal_len_sum * orientation_thr:
                head_inds = horizontal_edge_inds[0]
                tail_inds = horizontal_edge_inds[1]
            else:
                head_inds = vertical_edge_inds[0]
                tail_inds = vertical_edge_inds[1]

        return head_inds, tail_inds

    def _resample_line(self, line: ndarray, n: int) -> ndarray:
        """Resample n points on a line.

        Args:
            line (ndarray): The points composing a line.
            n (int): The resampled points number.

        Returns:
            resampled_line (ndarray): The points composing the resampled line.
        """

        assert line.ndim == 2
        assert line.shape[0] >= 2
        assert line.shape[1] == 2
        assert isinstance(n, int)
        assert n > 2

        edges_length, total_length = self._cal_curve_length(line)
        t_org = np.insert(np.cumsum(edges_length), 0, 0)
        unit_t = total_length / (n - 1)
        t_equidistant = np.arange(1, n - 1, dtype=np.float32) * unit_t
        edge_ind = 0
        points = [line[0]]
        for t in t_equidistant:
            while edge_ind < len(edges_length) - 1 and t > t_org[edge_ind + 1]:
                edge_ind += 1
            t_l, t_r = t_org[edge_ind], t_org[edge_ind + 1]
            weight = np.array([t_r - t, t - t_l], dtype=np.float32) / (
                t_r - t_l + self.eps)
            p_coords = np.dot(weight, line[[edge_ind, edge_ind + 1]])
            points.append(p_coords)
        points.append(line[-1])
        resampled_line = np.vstack(points)

        return resampled_line

    def _resample_sidelines(self, sideline1: ndarray, sideline2: ndarray,
                            resample_step: float) -> Tuple[ndarray, ndarray]:
        """Resample two sidelines to be of the same points number according to
        step size.

        Args:
            sideline1 (ndarray): The points composing a sideline of a text
                polygon.
            sideline2 (ndarray): The points composing another sideline of a
                text polygon.
            resample_step (float): The resampled step size.

        Returns:
            Tuple(resampled_line1, resampled_line2):

            - resampled_line1 (ndarray): The resampled line 1.
            - resampled_line2 (ndarray): The resampled line 2.
        """

        assert sideline1.ndim == sideline2.ndim == 2
        assert sideline1.shape[1] == sideline2.shape[1] == 2
        assert sideline1.shape[0] >= 2
        assert sideline2.shape[0] >= 2
        assert isinstance(resample_step, float)

        _, length1 = self._cal_curve_length(sideline1)
        _, length2 = self._cal_curve_length(sideline2)

        avg_length = (length1 + length2) / 2
        resample_point_num = max(int(float(avg_length) / resample_step) + 1, 3)

        resampled_line1 = self._resample_line(sideline1, resample_point_num)
        resampled_line2 = self._resample_line(sideline2, resample_point_num)

        return resampled_line1, resampled_line2

    def _cal_curve_length(self, line: ndarray) -> Tuple[ndarray, float]:
        """Calculate the length of each edge on the discrete curve and the sum.

        Args:
            line (ndarray): The points composing a discrete curve.

        Returns:
            Tuple(edges_length, total_length):

            - edge_length (ndarray): The length of each edge on the
              discrete curve.
            - total_length (float): The total length of the discrete
              curve.
        """

        assert line.ndim == 2
        assert len(line) >= 2

        edges_length = np.sqrt((line[1:, 0] - line[:-1, 0])**2 +
                               (line[1:, 1] - line[:-1, 1])**2)
        total_length = np.sum(edges_length)
        return edges_length, total_length

    def _draw_center_region_maps(self, top_line: ndarray, bot_line: ndarray,
                                 center_line: ndarray,
                                 center_region_mask: ndarray,
                                 radius_map: ndarray, sin_map: ndarray,
                                 cos_map: ndarray,
                                 region_shrink_ratio: float) -> None:
        """Draw attributes on text center region.

        Args:
            top_line (ndarray): The points composing top curved sideline of
                text polygon.
            bot_line (ndarray): The points composing bottom curved sideline
                of text polygon.
            center_line (ndarray): The points composing the center line of text
                instance.
            center_region_mask (ndarray): The text center region mask.
            radius_map (ndarray): The map where the distance from point to
                sidelines will be drawn on for each pixel in text center
                region.
            sin_map (ndarray): The map where vector_sin(theta) will be drawn
                on text center regions. Theta is the angle between tangent
                line and vector (1, 0).
            cos_map (ndarray): The map where vector_cos(theta) will be drawn on
                text center regions. Theta is the angle between tangent line
                and vector (1, 0).
            region_shrink_ratio (float): The shrink ratio of text center.
        """

        assert top_line.shape == bot_line.shape == center_line.shape
        assert (center_region_mask.shape == radius_map.shape == sin_map.shape
                == cos_map.shape)
        assert isinstance(region_shrink_ratio, float)
        for i in range(0, len(center_line) - 1):

            top_mid_point = (top_line[i] + top_line[i + 1]) / 2
            bot_mid_point = (bot_line[i] + bot_line[i + 1]) / 2
            radius = norm(top_mid_point - bot_mid_point) / 2

            text_direction = center_line[i + 1] - center_line[i]
            sin_theta = self.vector_sin(text_direction)
            cos_theta = self.vector_cos(text_direction)

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
            cv2.fillPoly(radius_map, [current_center_box], color=radius)

    def vector_angle(self, vec1: ndarray, vec2: ndarray) -> ndarray:
        """Compute the angle between two vectors."""
        if vec1.ndim > 1:
            unit_vec1 = vec1 / (norm(vec1, axis=-1) + self.eps).reshape(
                (-1, 1))
        else:
            unit_vec1 = vec1 / (norm(vec1, axis=-1) + self.eps)
        if vec2.ndim > 1:
            unit_vec2 = vec2 / (norm(vec2, axis=-1) + self.eps).reshape(
                (-1, 1))
        else:
            unit_vec2 = vec2 / (norm(vec2, axis=-1) + self.eps)
        return np.arccos(
            np.clip(np.sum(unit_vec1 * unit_vec2, axis=-1), -1.0, 1.0))

    def vector_slope(self, vec: ndarray) -> float:
        """Compute the slope of a vector."""
        assert len(vec) == 2
        return abs(vec[1] / (vec[0] + self.eps))

    def vector_sin(self, vec: ndarray) -> float:
        """Compute the sin of the angle between vector and x-axis."""
        assert len(vec) == 2
        return vec[1] / (norm(vec) + self.eps)

    def vector_cos(self, vec: ndarray) -> float:
        """Compute the cos of the angle between vector and x-axis."""
        assert len(vec) == 2
        return vec[0] / (norm(vec) + self.eps)
