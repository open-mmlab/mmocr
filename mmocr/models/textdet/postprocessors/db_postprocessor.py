# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence

import cv2
import numpy as np
import torch
from mmengine import InstanceData
from shapely.geometry import Polygon
from torch import Tensor

from mmocr.registry import MODELS
from mmocr.structures import TextDetDataSample
from mmocr.utils import offset_polygon
from .base_postprocessor import BaseTextDetPostProcessor


@MODELS.register_module()
class DBPostprocessor(BaseTextDetPostProcessor):
    """Decoding predictions of DbNet to instances. This is partially adapted
    from https://github.com/MhLiao/DB.

    Args:
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
            Defaults to 'poly'.
        rescale_fields (list[str]): The bbox/polygon field names to
            be rescaled. If None, no rescaling will be performed. Defaults to
            ['polygons'].
        mask_thr (float): The mask threshold value for binarization. Defaults
            to 0.3.
        min_text_score (float): The threshold value for converting binary map
            to shrink text regions. Defaults to 0.3.
        min_text_width (int): The minimum width of boundary polygon/box
            predicted. Defaults to 5.
        unclip_ratio (float): The unclip ratio for text regions dilation.
            Defaults to 1.5.
        epsilon_ratio (float): The epsilon ratio for approximation accuracy.
            Defaults to 0.01.
        max_candidates (int): The maximum candidate number. Defaults to 3000.
    """

    def __init__(self,
                 text_repr_type: str = 'poly',
                 rescale_fields: Sequence[str] = ['polygons'],
                 mask_thr: float = 0.3,
                 min_text_score: float = 0.3,
                 min_text_width: int = 5,
                 unclip_ratio: float = 1.5,
                 epsilon_ratio: float = 0.01,
                 max_candidates: int = 3000,
                 **kwargs) -> None:
        super().__init__(
            text_repr_type=text_repr_type,
            rescale_fields=rescale_fields,
            **kwargs)
        self.mask_thr = mask_thr
        self.min_text_score = min_text_score
        self.min_text_width = min_text_width
        self.unclip_ratio = unclip_ratio
        self.epsilon_ratio = epsilon_ratio
        self.max_candidates = max_candidates

    def __call__(self,
                 pred_results: List[Tensor],
                 data_samples: Sequence[TextDetDataSample],
                 training: bool = False) -> Sequence[TextDetDataSample]:
        """Postprocess pred_results according to metainfos in data_samples.

        Args:
            pred_results (Union[Tensor, List[Tensor]]): DBNet's prediction
                results as a list of tensor. The first tensor should be
                ``prob_logits`` of shape :math:`(N, H, W)`.
            data_samples (list[TextDetDataSample]): Batch of data_samples,
                each corresponding to a prediction result.
            training (bool): Whether the model is in training mode. Defaults to
                False.

        Returns:
            list[TextDetDataSample]: Batch of post-processed datasamples.
        """
        prob_maps = pred_results[0].sigmoid()
        return super().__call__(prob_maps, data_samples, training)

    def get_text_instances(self, prob_map: Tensor,
                           data_sample: TextDetDataSample
                           ) -> TextDetDataSample:
        """Get text instance predictions of one image.

        Args:
            pred_result (Tensor): DBNet's output ``prob_map`` of shape
                :math:`(H, W)`.
            data_sample (TextDetDataSample): Datasample of an image.

        Returns:
            TextDetDataSample: A new DataSample with predictions filled in.
            Polygons and results are saved in
            ``TextDetDataSample.pred_instances.polygons``. The confidence
            scores are saved in ``TextDetDataSample.pred_instances.scores``.
        """

        data_sample.pred_instances = InstanceData()
        data_sample.pred_instances.polygons = []
        data_sample.pred_instances.scores = []

        text_mask = prob_map > self.mask_thr

        score_map = prob_map.data.cpu().numpy().astype(np.float32)
        text_mask = text_mask.data.cpu().numpy().astype(np.uint8)  # to numpy

        contours, _ = cv2.findContours((text_mask * 255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for i, poly in enumerate(contours):
            if i > self.max_candidates:
                break
            epsilon = self.epsilon_ratio * cv2.arcLength(poly, True)
            approx = cv2.approxPolyDP(poly, epsilon, True)
            poly_pts = approx.reshape((-1, 2))
            if poly_pts.shape[0] < 4:
                continue
            score = self._get_bbox_score(score_map, poly_pts)
            if score < self.min_text_score:
                continue
            poly = self._unclip(poly_pts)
            # If the result polygon does not exist, or it is split into
            # multiple polygons, skip it.
            if len(poly) == 0:
                continue
            poly = poly.reshape(-1, 2)

            if self.text_repr_type == 'quad':
                rect = cv2.minAreaRect(poly)
                vertices = cv2.boxPoints(rect)
                poly = vertices.flatten() if min(
                    rect[1]) >= self.min_text_width else []
            elif self.text_repr_type == 'poly':
                poly = poly.flatten()

            if len(poly) < 8:
                poly = np.array([], dtype=np.float32)

            if len(poly) > 0:
                data_sample.pred_instances.polygons.append(poly)
                data_sample.pred_instances.scores.append(score)

        data_sample.pred_instances.scores = torch.FloatTensor(
            data_sample.pred_instances.scores)

        return data_sample

    def _get_bbox_score(self, score_map: np.ndarray,
                        poly_pts: np.ndarray) -> float:
        """Compute the average score over the area of the bounding box of the
        polygon.

        Args:
            score_map (np.ndarray): The score map.
            poly_pts (np.ndarray): The polygon points.

        Returns:
            float: The average score.
        """
        h, w = score_map.shape[:2]
        poly_pts = poly_pts.copy()
        xmin = np.clip(
            np.floor(poly_pts[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(
            np.ceil(poly_pts[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(
            np.floor(poly_pts[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(
            np.ceil(poly_pts[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        poly_pts[:, 0] = poly_pts[:, 0] - xmin
        poly_pts[:, 1] = poly_pts[:, 1] - ymin
        cv2.fillPoly(mask, poly_pts.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(score_map[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def _unclip(self, poly_pts: np.ndarray) -> np.ndarray:
        """Unclip a polygon.

        Args:
            poly_pts (np.ndarray): The polygon points.

        Returns:
            np.ndarray: The expanded polygon points.
        """
        poly = Polygon(poly_pts)
        distance = poly.area * self.unclip_ratio / poly.length
        return offset_polygon(poly_pts, distance)
