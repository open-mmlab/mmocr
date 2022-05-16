# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence

import torch
from mmcv.runner import auto_fp16
from mmdet.models.detectors.base import BaseDetector as MMDET_BaseDetector

from mmocr.core.data_structures import TextDetDataSample
from mmocr.registry import MODELS


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
        preprocess_cfg (dict, optional): Model preprocessing config
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
                 preprocess_cfg: Optional[Dict] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(preprocess_cfg=preprocess_cfg, init_cfg=init_cfg)
        assert det_head is not None, 'det_head cannot be None!'
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.det_head = MODELS.build(det_head)

    def extract_feat(self, img: torch.Tensor) -> torch.Tensor:
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img: torch.Tensor,
                      data_samples: Sequence[TextDetDataSample]) -> Dict:
        """
        Args:
            img (torch.Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            data_samples (list[TextDetDataSample]): A list of N datasamples,
                containing meta information and gold annotations for each of
                the images.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img)
        preds = self.det_head(x, data_samples)
        losses = self.det_head.loss(preds, data_samples)
        return losses

    def simple_test(self, img: torch.Tensor,
                    data_samples: Sequence[TextDetDataSample]
                    ) -> Sequence[TextDetDataSample]:
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images of shape (N, C, H, W).
            data_samples (list[TextDetDataSample]): A list of N datasamples,
                containing meta information and gold annotations for each of
                the images.

        Returns:
            list[TextDetDataSample]: A list of N datasamples of prediction
            results. Results are stored in ``pred_instances``.
        """
        x = self.extract_feat(img)
        preds = self.det_head(x, data_samples)
        return self.det_head.postprocessor(preds, data_samples)

    def aug_test(
        self, imgs: Sequence[torch.Tensor],
        data_samples: Sequence[Sequence[TextDetDataSample]]
    ) -> Sequence[Sequence[TextDetDataSample]]:
        """Test function with test time augmentation."""
        raise NotImplementedError

    @auto_fp16(apply_to=('imgs', ))
    def forward_simple_test(self, imgs: torch.Tensor,
                            data_samples: Sequence[TextDetDataSample]
                            ) -> Sequence[TextDetDataSample]:
        """Test forward function called by self.forward() when running in test
        mode without test time augmentation.

        Though not useful in MMOCR, it has been kept to maintain the maximum
        compatibility with MMDetection's BaseDetector.

        Args:
            img (torch.Tensor): Images of shape (N, C, H, W).
            data_samples (list[TextDetDataSample]): A list of N datasamples,
                containing meta information and gold annotations for each of
                the images.

        Returns:
            list[TextDetDataSample]: A list of N datasamples of prediction
            results. Results are stored in ``pred_instances``.
        """
        return self.simple_test(imgs, data_samples)
