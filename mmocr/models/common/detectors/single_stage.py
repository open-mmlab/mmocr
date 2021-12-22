# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmdet.models.detectors import \
    SingleStageDetector as MMDET_SingleStageDetector

from mmocr.models.builder import (DETECTORS, build_backbone, build_head,
                                  build_neck)


@DETECTORS.register_module()
class SingleStageDetector(MMDET_SingleStageDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(MMDET_SingleStageDetector, self).__init__(init_cfg=init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
