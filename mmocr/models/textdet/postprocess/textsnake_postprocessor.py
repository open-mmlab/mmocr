# Copyright (c) OpenMMLab. All rights reserved.

import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize

from mmocr.models.builder import POSTPROCESSOR
from .base_postprocessor import BasePostprocessor
from .wrapper import centralize, fill_hole, merge_disks


@POSTPROCESSOR.register_module()
class TextSnakePostprocessor(BasePostprocessor):
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
    """

    def __init__(self,
                 text_repr_type='poly',
                 min_text_region_confidence=0.6,
                 min_center_region_confidence=0.2,
                 min_center_area=30,
                 disk_overlap_thr=0.03,
                 radius_shrink_ratio=1.03,
                 **kwargs):
        super().__init__(text_repr_type)
        assert text_repr_type == 'poly'
        self.min_text_region_confidence = min_text_region_confidence
        self.min_center_region_confidence = min_center_region_confidence
        self.min_center_area = min_center_area
        self.disk_overlap_thr = disk_overlap_thr
        self.radius_shrink_ratio = radius_shrink_ratio

    def __call__(self, preds):
        """
        Args:
            preds (Tensor): Prediction map with shape :math:`(C, H, W)`.

        Returns:
            list[list[float]]: The instance boundary and its confidence.
        """
        assert preds.dim() == 3

        preds[:2, :, :] = torch.sigmoid(preds[:2, :, :])
        preds = preds.detach().cpu().numpy()

        pred_text_score = preds[0]
        pred_text_mask = pred_text_score > self.min_text_region_confidence
        pred_center_score = preds[1] * pred_text_score
        pred_center_mask = \
            pred_center_score > self.min_center_region_confidence
        pred_sin = preds[2]
        pred_cos = preds[3]
        pred_radius = preds[4]
        mask_sz = pred_text_mask.shape

        scale = np.sqrt(1.0 / (pred_sin**2 + pred_cos**2 + 1e-8))
        pred_sin = pred_sin * scale
        pred_cos = pred_cos * scale

        pred_center_mask = fill_hole(pred_center_mask).astype(np.uint8)
        center_contours, _ = cv2.findContours(pred_center_mask, cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)

        boundaries = []
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

            center_line_yx = centralize(skeleton_yx, cos, -sin, radius,
                                        instance_center_mask)
            y, x = center_line_yx[:, 0], center_line_yx[:, 1]
            radius = (pred_radius[y, x] * self.radius_shrink_ratio).reshape(
                (-1, 1))
            score = pred_center_score[y, x].reshape((-1, 1))
            instance_disks = np.hstack(
                [np.fliplr(center_line_yx), radius, score])
            instance_disks = merge_disks(instance_disks, self.disk_overlap_thr)

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
                boundary = contours[0].flatten().tolist()
                boundaries.append(boundary + [score])

        return boundaries
