# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence

import cv2
import numpy as np
import torch
from mmcv.ops import pixel_group
from mmengine.structures import InstanceData

from mmocr.registry import MODELS
from mmocr.structures import TextDetDataSample
from .base import BaseTextDetPostProcessor


@MODELS.register_module()
class PANPostprocessor(BaseTextDetPostProcessor):
    """Convert scores to quadrangles via post processing in PANet. This is
    partially adapted from https://github.com/WenmuZhou/PAN.pytorch.

    Args:
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
            Defaults to 'poly'.
        score_threshold (float): The minimal text score.
            Defaults to 0.3.
        rescale_fields (list[str]): The bbox/polygon field names to
            be rescaled. If None, no rescaling will be performed. Defaults to
            ['polygons'].
        min_text_confidence (float): The minimal text confidence.
            Defaults to 0.5.
        min_kernel_confidence (float): The minimal kernel confidence.
            Defaults to 0.5.
        distance_threshold (float): The minimal distance between the point to
            mean of text kernel. Defaults to 3.0.
        min_text_area (int): The minimal text instance region area.
            Defaults to 16.
        downsample_ratio (float): Downsample ratio. Defaults to 0.25.
    """

    def __init__(self,
                 text_repr_type: str = 'poly',
                 score_threshold: float = 0.3,
                 rescale_fields: Sequence[str] = ['polygons'],
                 min_text_confidence: float = 0.5,
                 min_kernel_confidence: float = 0.5,
                 distance_threshold: float = 3.0,
                 min_text_area: int = 16,
                 downsample_ratio: float = 0.25) -> None:
        super().__init__(text_repr_type, rescale_fields)

        self.min_text_confidence = min_text_confidence
        self.min_kernel_confidence = min_kernel_confidence
        self.score_threshold = score_threshold
        self.min_text_area = min_text_area
        self.distance_threshold = distance_threshold
        self.downsample_ratio = downsample_ratio

    def get_text_instances(self, pred_results: torch.Tensor,
                           data_sample: TextDetDataSample,
                           **kwargs) -> TextDetDataSample:
        """Get text instance predictions of one image.

        Args:
            pred_result (torch.Tensor): Prediction results of an image which
                is a tensor of shape :math:`(N, H, W)`.
            data_sample (TextDetDataSample): Datasample of an image.

        Returns:
            TextDetDataSample: A new DataSample with predictions filled in.
            Polygons and results are saved in
            ``TextDetDataSample.pred_instances.polygons``. The confidence
            scores are saved in ``TextDetDataSample.pred_instances.scores``.
        """
        assert pred_results.dim() == 3

        pred_results[:2, :, :] = torch.sigmoid(pred_results[:2, :, :])
        pred_results = pred_results.detach().cpu().numpy()

        text_score = pred_results[0].astype(np.float32)
        text = pred_results[0] > self.min_text_confidence
        kernel = (pred_results[1] > self.min_kernel_confidence) * text
        embeddings = pred_results[2:] * text.astype(np.float32)
        embeddings = embeddings.transpose((1, 2, 0))  # (h, w, 4)

        region_num, labels = cv2.connectedComponents(
            kernel.astype(np.uint8), connectivity=4)
        contours, _ = cv2.findContours((kernel * 255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        kernel_contours = np.zeros(text.shape, dtype='uint8')
        cv2.drawContours(kernel_contours, contours, -1, 255)
        text_points = pixel_group(text_score, text, embeddings, labels,
                                  kernel_contours, region_num,
                                  self.distance_threshold)

        polygons = []
        scores = []
        for text_point in text_points:
            text_confidence = text_point[0]
            text_point = text_point[2:]
            text_point = np.array(text_point, dtype=int).reshape(-1, 2)
            area = text_point.shape[0]
            if (area < self.min_text_area
                    or text_confidence <= self.score_threshold):
                continue

            polygon = self._points2boundary(text_point)
            if len(polygon) > 0:
                polygons.append(polygon)
                scores.append(text_confidence)
        pred_instances = InstanceData()
        pred_instances.polygons = polygons
        pred_instances.scores = torch.FloatTensor(scores)
        data_sample.pred_instances = pred_instances
        scale_factor = data_sample.scale_factor
        scale_factor = tuple(factor * self.downsample_ratio
                             for factor in scale_factor)
        data_sample.set_metainfo(dict(scale_factor=scale_factor))
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

    def _points2boundary(self,
                         points: np.ndarray,
                         min_width: int = 0) -> List[float]:
        """Convert a text mask represented by point coordinates sequence into a
        text boundary.

        Args:
            points (ndarray): Mask index of size (n, 2).
            min_width (int): Minimum bounding box width to be converted. Only
                applicable to 'quad' type. Defaults to 0.

        Returns:
            list[float]: The text boundary point coordinates (x, y) list.
            Return [] if no text boundary found.
        """
        assert isinstance(points, np.ndarray)
        assert points.shape[1] == 2
        assert self.text_repr_type in ['quad', 'poly']

        if self.text_repr_type == 'quad':
            rect = cv2.minAreaRect(points)
            vertices = cv2.boxPoints(rect)
            boundary = []
            if min(rect[1]) >= min_width:
                boundary = [p for p in vertices.flatten().tolist()]
        elif self.text_repr_type == 'poly':

            height = np.max(points[:, 1]) + 10
            width = np.max(points[:, 0]) + 10

            mask = np.zeros((height, width), np.uint8)
            mask[points[:, 1], points[:, 0]] = 255

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            boundary = list(contours[0].flatten().tolist())

        if len(boundary) < 8:
            return []

        return boundary
