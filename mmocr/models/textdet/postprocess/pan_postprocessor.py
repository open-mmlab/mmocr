# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
import torch
from mmcv.ops import pixel_group

from mmocr.core import points2boundary
from mmocr.models.builder import POSTPROCESSOR
from .base_postprocessor import BasePostprocessor


@POSTPROCESSOR.register_module()
class PANPostprocessor(BasePostprocessor):
    """Convert scores to quadrangles via post processing in PANet. This is
    partially adapted from https://github.com/WenmuZhou/PAN.pytorch.

    Args:
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
        min_text_confidence (float): The minimal text confidence.
        min_kernel_confidence (float): The minimal kernel confidence.
        min_text_avg_confidence (float): The minimal text average confidence.
        min_text_area (int): The minimal text instance region area.

    Returns:
        list[list[float]]: The instance boundary and its confidence.
    """

    def __init__(self,
                 text_repr_type='poly',
                 min_text_confidence=0.5,
                 min_kernel_confidence=0.5,
                 min_text_avg_confidence=0.85,
                 min_text_area=16):
        super().__init__(text_repr_type)

        self.min_text_confidence = min_text_confidence
        self.min_kernel_confidence = min_kernel_confidence
        self.min_text_avg_confidence = min_text_avg_confidence
        self.min_text_area = min_text_area

    def __call__(self, preds):
        assert preds.dim() == 3

        preds[:2, :, :] = torch.sigmoid(preds[:2, :, :])
        preds = preds.detach().cpu().numpy()

        text_score = preds[0].astype(np.float32)
        text = preds[0] > self.min_text_confidence
        kernel = (preds[1] > self.min_kernel_confidence) * text
        embeddings = preds[2:].transpose((1, 2, 0))  # (h, w, 4)

        region_num, labels = cv2.connectedComponents(
            kernel.astype(np.uint8), connectivity=4)
        contours, _ = cv2.findContours((kernel * 255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        kernel_contours = np.zeros(text.shape, dtype='uint8')
        cv2.drawContours(kernel_contours, contours, -1, 255)
        text_points = pixel_group(text_score, text, embeddings, labels,
                                  kernel_contours, region_num,
                                  self.min_text_avg_confidence)

        boundaries = []
        for text_point in text_points:
            text_confidence = text_point[0]
            text_point = text_point[2:]
            text_point = np.array(text_point, dtype=int).reshape(-1, 2)
            area = text_point.shape[0]

            if not self.is_valid_instance(area, text_confidence,
                                          self.min_text_area,
                                          self.min_text_avg_confidence):
                continue

            vertices_confidence = points2boundary(text_point,
                                                  self.text_repr_type,
                                                  text_confidence)
            if vertices_confidence is not None:
                boundaries.append(vertices_confidence)

        return boundaries
