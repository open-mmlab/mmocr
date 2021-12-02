# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np

from mmocr.core import points2boundary
from mmocr.models.builder import POSTPROCESSOR
from .base_postprocessor import BasePostprocessor
from .wrapper import box_score_fast, unclip


@POSTPROCESSOR.register_module()
class DBPostprocessor(BasePostprocessor):
    """Decoding predictions of DbNet to instances. This is partially adapted
    from https://github.com/MhLiao/DB.

    Args:
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
        mask_thr (float): The mask threshold value for binarization.
        min_text_score (float): The threshold value for converting binary map
            to shrink text regions.
        min_text_width (int): The minimum width of boundary polygon/box
            predicted.
        unclip_ratio (float): The unclip ratio for text regions dilation.
        max_candidates (int): The maximum candidate number.
    """

    def __init__(self,
                 text_repr_type='poly',
                 mask_thr=0.3,
                 min_text_score=0.3,
                 min_text_width=5,
                 unclip_ratio=1.5,
                 max_candidates=3000):
        super().__init__(text_repr_type)
        self.mask_thr = mask_thr
        self.min_text_score = min_text_score
        self.min_text_width = min_text_width
        self.unclip_ratio = unclip_ratio
        self.max_candidates = max_candidates

    def __call__(self, preds):
        """
        Args:
            preds (Tensor): Prediction map with shape :math:`(C, H, W)`.

        Returns:
            list[list[float]]: The predicted text boundaries.
        """
        assert preds.dim() == 3

        prob_map = preds[0, :, :]
        text_mask = prob_map > self.mask_thr

        score_map = prob_map.data.cpu().numpy().astype(np.float32)
        text_mask = text_mask.data.cpu().numpy().astype(np.uint8)  # to numpy

        contours, _ = cv2.findContours((text_mask * 255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boundaries = []
        for i, poly in enumerate(contours):
            if i > self.max_candidates:
                break
            epsilon = 0.01 * cv2.arcLength(poly, True)
            approx = cv2.approxPolyDP(poly, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = box_score_fast(score_map, points)
            if score < self.min_text_score:
                continue
            poly = unclip(points, unclip_ratio=self.unclip_ratio)
            if len(poly) == 0 or isinstance(poly[0], list):
                continue
            poly = poly.reshape(-1, 2)

            if self.text_repr_type == 'quad':
                poly = points2boundary(poly, self.text_repr_type, score,
                                       self.min_text_width)
            elif self.text_repr_type == 'poly':
                poly = poly.flatten().tolist()
                if score is not None:
                    poly = poly + [score]
                if len(poly) < 8:
                    poly = None

            if poly is not None:
                boundaries.append(poly)

        return boundaries
