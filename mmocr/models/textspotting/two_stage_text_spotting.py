# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.models import BaseTextDetector
from mmocr.registry import MODELS
from mmocr.structures import TextDetDataSample  # noqa: F401
from mmocr.structures import TextRecogDataSample  # noqa: F401
from mmocr.structures import TextSpottingDataSample  # noqa: F401
from mmocr.utils import det_to_spotting, spotting_to_det
from mmocr.utils.typing import OptConfigType, OptMultiConfig


class TwoStageTextSpotter(BaseTextDetector):

    def __init__(self,
                 backbone=None,
                 neck=None,
                 det_head=None,
                 roi_head=None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):

        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        if det_head is not None:
            self.det_head = MODELS.build(det_head)

        if roi_head is not None:
            self.roi_head = MODELS.build(roi_head)

    @property
    def with_det_head(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'det_head') and self.det_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def loss(self, inputs, data_samples):
        losses = dict()
        inputs = self.extract_feat(inputs)
        det_data_samples = [
            TextDetDataSample(gt_instances=ds.gt_instances)
            for ds in data_samples
        ]
        det_loss, det_preds = self.det_head.loss_and_predict(
            inputs, det_data_samples)
        data_samples = det_to_spotting(det_preds, data_samples)
        losses.update(det_loss)

        roi_losses = self.roi_head.loss(inputs, data_samples)
        losses.update(roi_losses)
        return losses

    def predict(self, inputs, data_samples):
        inputs = self.extract_feat(inputs)
        det_data_samples = spotting_to_det(data_samples)
        det_preds = self.det_head.predict(inputs, det_data_samples)
        data_samples = det_to_spotting(det_preds, data_samples)
        recog_preds = self.roi_head.predict(inputs, data_samples)
        return recog_preds
