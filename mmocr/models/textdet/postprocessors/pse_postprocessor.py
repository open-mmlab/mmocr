# Copyright (c) OpenMMLab. All rights reserved.

from typing import List

import cv2
import numpy as np
import torch
from mmcv.ops import contour_expand
from mmengine.structures import InstanceData

from mmocr.registry import MODELS
from mmocr.structures import TextDetDataSample
from .pan_postprocessor import PANPostprocessor


@MODELS.register_module()
class PSEPostprocessor(PANPostprocessor):
    """Decoding predictions of PSENet to instances. This is partially adapted
    from https://github.com/whai362/PSENet.

    Args:
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
            Defaults to 'poly'.
        rescale_fields (list[str]): The bbox/polygon field names to
            be rescaled. If None, no rescaling will be performed. Defaults to
            ['polygons'].
        min_kernel_confidence (float): The minimal kernel confidence.
            Defaults to 0.5.
        score_threshold (float): The minimal text average confidence.
            Defaults to 0.3.
        min_kernel_area (int): The minimal text kernel area. Defaults to 0.
        min_text_area (int): The minimal text instance region area.
            Defaults to 16.
        downsample_ratio (float): Downsample ratio. Defaults to 0.25.
    """

    def __init__(self,
                 text_repr_type: str = 'poly',
                 rescale_fields: List[str] = ['polygons'],
                 min_kernel_confidence: float = 0.5,
                 score_threshold: float = 0.3,
                 min_kernel_area: int = 0,
                 min_text_area: int = 16,
                 downsample_ratio: float = 0.25) -> None:
        super().__init__(
            text_repr_type=text_repr_type,
            rescale_fields=rescale_fields,
            min_kernel_confidence=min_kernel_confidence,
            score_threshold=score_threshold,
            min_text_area=min_text_area,
            downsample_ratio=downsample_ratio)
        self.min_kernel_area = min_kernel_area

    def get_text_instances(self, pred_results: torch.Tensor,
                           data_sample: TextDetDataSample,
                           **kwargs) -> TextDetDataSample:
        """
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

        pred_results = torch.sigmoid(pred_results)  # text confidence

        masks = pred_results > self.min_kernel_confidence
        text_mask = masks[0, :, :]
        kernel_masks = masks[0:, :, :] * text_mask
        kernel_masks = kernel_masks.data.cpu().numpy().astype(np.uint8)

        score = pred_results[0, :, :]
        score = score.data.cpu().numpy().astype(np.float32)

        region_num, labels = cv2.connectedComponents(
            kernel_masks[-1], connectivity=4)

        labels = contour_expand(kernel_masks, labels, self.min_kernel_area,
                                region_num)
        labels = np.array(labels)
        label_num = np.max(labels)

        polygons = []
        scores = []
        for i in range(1, label_num + 1):
            points = np.array(np.where(labels == i)).transpose((1, 0))[:, ::-1]
            area = points.shape[0]
            score_instance = np.mean(score[labels == i])
            if not (area >= self.min_text_area
                    or score_instance > self.score_threshold):
                continue

            polygon = self._points2boundary(points)
            if polygon:
                polygons.append(polygon)
                scores.append(score_instance)

        pred_instances = InstanceData()
        pred_instances.polygons = polygons
        pred_instances.scores = torch.FloatTensor(scores)
        data_sample.pred_instances = pred_instances
        scale_factor = data_sample.scale_factor
        scale_factor = tuple(factor * self.downsample_ratio
                             for factor in scale_factor)
        data_sample.set_metainfo(dict(scale_factor=scale_factor))

        return data_sample
