# Copyright (c) OpenMMLab. All rights reserved.

from typing import List, Sequence

import cv2
import numpy as np
import torch
from mmengine.structures import InstanceData
from numpy.linalg import norm
from skimage.morphology import skeletonize

from mmengine.registry import MODELS
from mmocr.structures import TextDetDataSample
from mmocr.utils import fill_hole
from .base import BaseTextDetPostProcessor


@MODELS.register_module()
class TextSnakePostprocessor(BaseTextDetPostProcessor):
    """Decoding predictions of TextSnake to instances. This was partially
    adapted from https://github.com/princewang1994/TextSnake.pytorch.

    Args:
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
        min_text_region_confidence (float): The confidence threshold of text
            region in TextSnake.
        min_center_region_confidence (float): The confidence threshold of text
            center region in TextSnake.
        min_center_area (int): The minimal text center region area.
        disk_overlap_thr (float): The radius overlap threshold for merging
            disks.
        radius_shrink_ratio (float): The shrink ratio of ordered disks radii.
        rescale_fields (list[str], optional): The bbox/polygon field names to
            be rescaled. If None, no rescaling will be performed.
    """

    def __init__(self,
                 text_repr_type: str = 'poly',
                 min_text_region_confidence: float = 0.6,
                 min_center_region_confidence: float = 0.2,
                 min_center_area: int = 30,
                 disk_overlap_thr: float = 0.03,
                 radius_shrink_ratio: float = 1.03,
                 rescale_fields: Sequence[str] = ['polygons'],
                 **kwargs) -> None:
        super().__init__(
            text_repr_type=text_repr_type,
            rescale_fields=rescale_fields,
            **kwargs)
        assert text_repr_type == 'poly'
        self.min_text_region_confidence = min_text_region_confidence
        self.min_center_region_confidence = min_center_region_confidence
        self.min_center_area = min_center_area
        self.disk_overlap_thr = disk_overlap_thr
        self.radius_shrink_ratio = radius_shrink_ratio

    def get_text_instances(self, pred_results: torch.Tensor,
                           data_sample: TextDetDataSample
                           ) -> TextDetDataSample:
        """
        Args:
            pred_results (torch.Tensor): Prediction map with
                shape :math:`(C, H, W)`.
            data_sample (TextDetDataSample): Datasample of an image.

        Returns:
            list[list[float]]: The instance boundary and its confidence.
        """
        assert pred_results.dim() == 3
        data_sample.pred_instances = InstanceData()
        data_sample.pred_instances.polygons = []
        data_sample.pred_instances.scores = []

        pred_results[:2, :, :] = torch.sigmoid(pred_results[:2, :, :])
        pred_results = pred_results.detach().cpu().numpy()

        pred_text_score = pred_results[0]
        pred_text_mask = pred_text_score > self.min_text_region_confidence
        pred_center_score = pred_results[1] * pred_text_score
        pred_center_mask = \
            pred_center_score > self.min_center_region_confidence
        pred_sin = pred_results[2]
        pred_cos = pred_results[3]
        pred_radius = pred_results[4]
        mask_sz = pred_text_mask.shape

        scale = np.sqrt(1.0 / (pred_sin**2 + pred_cos**2 + 1e-8))
        pred_sin = pred_sin * scale
        pred_cos = pred_cos * scale

        pred_center_mask = fill_hole(pred_center_mask).astype(np.uint8)
        center_contours, _ = cv2.findContours(pred_center_mask, cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)

        for contour in center_contours:
            if cv2.contourArea(contour) < self.min_center_area:
                continue
            instance_center_mask = np.zeros(mask_sz, dtype=np.uint8)
            cv2.drawContours(instance_center_mask, [contour], -1, 1, -1)
            skeleton = skeletonize(instance_center_mask)
            skeleton_yx = np.argwhere(skeleton > 0)
            y, x = skeleton_yx[:, 0], skeleton_yx[:, 1]
            cos = pred_cos[y, x].reshape((-1, 1))
            sin = pred_sin[y, x].reshape((-1, 1))
            radius = pred_radius[y, x].reshape((-1, 1))

            center_line_yx = self._centralize(skeleton_yx, cos, -sin, radius,
                                              instance_center_mask)
            y, x = center_line_yx[:, 0], center_line_yx[:, 1]
            radius = (pred_radius[y, x] * self.radius_shrink_ratio).reshape(
                (-1, 1))
            score = pred_center_score[y, x].reshape((-1, 1))
            instance_disks = np.hstack(
                [np.fliplr(center_line_yx), radius, score])
            instance_disks = self._merge_disks(instance_disks,
                                               self.disk_overlap_thr)

            instance_mask = np.zeros(mask_sz, dtype=np.uint8)
            for x, y, radius, score in instance_disks:
                if radius > 1:
                    cv2.circle(instance_mask, (int(x), int(y)), int(radius), 1,
                               -1)
            contours, _ = cv2.findContours(instance_mask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

            score = np.sum(instance_mask * pred_text_score) / (
                np.sum(instance_mask) + 1e-8)
            if (len(contours) > 0 and cv2.contourArea(contours[0]) > 0
                    and contours[0].size > 8):
                polygon = contours[0].flatten().tolist()
                data_sample.pred_instances.polygons.append(polygon)
                data_sample.pred_instances.scores.append(score)

        data_sample.pred_instances.scores = torch.FloatTensor(
            data_sample.pred_instances.scores)

        return data_sample

    def split_results(self, pred_results: torch.Tensor) -> List[torch.Tensor]:
        """Split the prediction results into text score and kernel score.

        Args:
            pred_results (torch.Tensor): The prediction results.

        Returns:
            List[torch.Tensor]: The text score and kernel score.
        """
        pred_results = [pred_result for pred_result in pred_results]
        return pred_results

    @staticmethod
    def _centralize(points_yx: np.ndarray,
                    normal_cos: torch.Tensor,
                    normal_sin: torch.Tensor,
                    radius: torch.Tensor,
                    contour_mask: np.ndarray,
                    step_ratio: float = 0.03) -> np.ndarray:
        """Centralize the points.

        Args:
            points_yx (np.array): The points in yx order.
            normal_cos (torch.Tensor): The normal cosine of the points.
            normal_sin (torch.Tensor): The normal sine of the points.
            radius (torch.Tensor): The radius of the points.
            contour_mask (np.array): The contour mask of the points.
            step_ratio (float): The step ratio of the centralization.
                Defaults to 0.03.

        Returns:
            np.ndarray: The centralized points.
        """

        h, w = contour_mask.shape
        top_yx = bot_yx = points_yx
        step_flags = np.ones((len(points_yx), 1), dtype=np.bool_)
        step = step_ratio * radius * np.hstack([normal_cos, normal_sin])
        while np.any(step_flags):
            next_yx = np.array(top_yx + step, dtype=np.int32)
            next_y, next_x = next_yx[:, 0], next_yx[:, 1]
            step_flags = (next_y >= 0) & (next_y < h) & (next_x > 0) & (
                next_x < w) & contour_mask[np.clip(next_y, 0, h - 1),
                                           np.clip(next_x, 0, w - 1)]
            top_yx = top_yx + step_flags.reshape((-1, 1)) * step
        step_flags = np.ones((len(points_yx), 1), dtype=np.bool_)
        while np.any(step_flags):
            next_yx = np.array(bot_yx - step, dtype=np.int32)
            next_y, next_x = next_yx[:, 0], next_yx[:, 1]
            step_flags = (next_y >= 0) & (next_y < h) & (next_x > 0) & (
                next_x < w) & contour_mask[np.clip(next_y, 0, h - 1),
                                           np.clip(next_x, 0, w - 1)]
            bot_yx = bot_yx - step_flags.reshape((-1, 1)) * step
        centers = np.array((top_yx + bot_yx) * 0.5, dtype=np.int32)
        return centers

    @staticmethod
    def _merge_disks(disks: np.ndarray, disk_overlap_thr: float) -> np.ndarray:
        """Merging overlapped disks.

        Args:
            disks (np.ndarray): The predicted disks.
            disk_overlap_thr (float): The radius overlap threshold for merging
                disks.

        Returns:
            np.ndarray: The merged disks.
        """
        xy = disks[:, 0:2]
        radius = disks[:, 2]
        scores = disks[:, 3]
        order = scores.argsort()[::-1]

        merged_disks = []
        while order.size > 0:
            if order.size == 1:
                merged_disks.append(disks[order])
                break
            i = order[0]
            d = norm(xy[i] - xy[order[1:]], axis=1)
            ri = radius[i]
            r = radius[order[1:]]
            d_thr = (ri + r) * disk_overlap_thr

            merge_inds = np.where(d <= d_thr)[0] + 1
            if merge_inds.size > 0:
                merge_order = np.hstack([i, order[merge_inds]])
                merged_disks.append(np.mean(disks[merge_order], axis=0))
            else:
                merged_disks.append(disks[i])

            inds = np.where(d > d_thr)[0] + 1
            order = order[inds]
        merged_disks = np.vstack(merged_disks)

        return merged_disks
