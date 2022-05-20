# Copyright (c) OpenMMLab. All rights reserved.

import cv2
import numpy as np
import torch
from mmcv.ops import contour_expand

from mmocr.core import points2boundary
from mmocr.registry import MODELS
from .base_postprocessor import BasePostprocessor


@MODELS.register_module()
class PSEPostprocessor(BasePostprocessor):
    """Decoding predictions of PSENet to instances. This is partially adapted
    from https://github.com/whai362/PSENet.

    Args:
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
        min_kernel_confidence (float): The minimal kernel confidence.
        min_text_avg_confidence (float): The minimal text average confidence.
        min_kernel_area (int): The minimal text kernel area.
        min_text_area (int): The minimal text instance region area.
    """

    def __init__(self,
                 text_repr_type='poly',
                 min_kernel_confidence=0.5,
                 min_text_avg_confidence=0.85,
                 min_kernel_area=0,
                 min_text_area=16,
                 **kwargs):
        super().__init__(text_repr_type)

        assert 0 <= min_kernel_confidence <= 1
        assert 0 <= min_text_avg_confidence <= 1
        assert isinstance(min_kernel_area, int)
        assert isinstance(min_text_area, int)

        self.min_kernel_confidence = min_kernel_confidence
        self.min_text_avg_confidence = min_text_avg_confidence
        self.min_kernel_area = min_kernel_area
        self.min_text_area = min_text_area

    def __call__(self, preds):
        """
        Args:
            preds (Tensor): Prediction map with shape :math:`(C, H, W)`.

        Returns:
            list[list[float]]: The instance boundary and its confidence.
        """
        assert preds.dim() == 3

        preds = torch.sigmoid(preds)  # text confidence

        score = preds[0, :, :]
        masks = preds > self.min_kernel_confidence
        text_mask = masks[0, :, :]
        kernel_masks = masks[0:, :, :] * text_mask

        score = score.data.cpu().numpy().astype(np.float32)

        kernel_masks = kernel_masks.data.cpu().numpy().astype(np.uint8)

        region_num, labels = cv2.connectedComponents(
            kernel_masks[-1], connectivity=4)

        labels = contour_expand(kernel_masks, labels, self.min_kernel_area,
                                region_num)
        labels = np.array(labels)
        label_num = np.max(labels)
        boundaries = []
        for i in range(1, label_num + 1):
            points = np.array(np.where(labels == i)).transpose((1, 0))[:, ::-1]
            area = points.shape[0]
            score_instance = np.mean(score[labels == i])
            if not self.is_valid_instance(area, score_instance,
                                          self.min_text_area,
                                          self.min_text_avg_confidence):
                continue

            vertices_confidence = points2boundary(points, self.text_repr_type,
                                                  score_instance)
            if vertices_confidence is not None:
                boundaries.append(vertices_confidence)

        return boundaries
