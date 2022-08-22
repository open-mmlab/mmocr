# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence

import torch
from mmdet.models.detectors.base import BaseDetector as MMDET_BaseDetector

from mmocr.registry import MODELS
from mmocr.structures import TextDetDataSample


@MODELS.register_module()
class SingleStageTextDetector(MMDET_BaseDetector):
    """The class for implementing single stage text detector.

    Single-stage text detectors directly and densely predict bounding boxes or
    polygons on the output features of the backbone + neck (optional).

    Args:
        backbone (dict): Backbone config.
        neck (dict, optional): Neck config. If None, the output from backbone
            will be directly fed into ``det_head``.
        det_head (dict): Head config.
        data_preprocessor (dict, optional): Model preprocessing config
            for processing the input image data. Keys allowed are
            ``to_rgb``(bool), ``pad_size_divisor``(int), ``pad_value``(int or
            float), ``mean``(int or float) and ``std``(int or float).
            Preprcessing order: 1. to rgb; 2. normalization 3. pad.
            Defaults to None.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 backbone: Dict,
                 det_head: Dict,
                 neck: Optional[Dict] = None,
                 data_preprocessor: Optional[Dict] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        assert det_head is not None, 'det_head cannot be None!'
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.det_head = MODELS.build(det_head)

    def extract_feat(self, inputs: torch.Tensor) -> torch.Tensor:
        """Extract features.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            Tensor or tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        inputs = self.backbone(inputs)
        if self.with_neck:
            inputs = self.neck(inputs)
        return inputs

    def loss(self, inputs: torch.Tensor,
             data_samples: Sequence[TextDetDataSample]) -> Dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            data_samples (list[TextDetDataSample]): A list of N
                datasamples, containing meta information and gold annotations
                for each of the images.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        inputs = self.extract_feat(inputs)
        return self.det_head.loss(inputs, data_samples)

    def predict(self, inputs: torch.Tensor,
                data_samples: Sequence[TextDetDataSample]
                ) -> Sequence[TextDetDataSample]:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (torch.Tensor): Images of shape (N, C, H, W).
            data_samples (list[TextDetDataSample]): A list of N
                datasamples, containing meta information and gold annotations
                for each of the images.

        Returns:
            list[TextDetDataSample]: A list of N datasamples of prediction
            results.  Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - polygons (list[np.ndarray]): The length is num_instances.
                    Each element represents the polygon of the
                    instance, in (xn, yn) order.
        """
        x = self.extract_feat(inputs)
        return self.det_head.predict(x, data_samples)

    def _forward(self,
                 inputs: torch.Tensor,
                 data_samples: Optional[Sequence[TextDetDataSample]] = None,
                 **kwargs) -> torch.Tensor:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (list[TextDetDataSample]): A list of N
                datasamples, containing meta information and gold annotations
                for each of the images.

        Returns:
            Tensor or tuple[Tensor]: A tuple of features from ``det_head``
            forward.
        """
        x = self.extract_feat(inputs)
        return self.det_head(x, data_samples)
