# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models import BaseDetector

from ..builder import build_backbone, build_head, build_neck


class TwoStageTextSpotter(BaseDetector):

    def __init__(self,
                 backbone=None,
                 neck=None,
                 detect_head=None,
                 recognition_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(TwoStageTextSpotter, self).__init__(init_cfg)

        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if detect_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            detect_head_ = detect_head.copy()
            detect_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.detect_head = build_head(detect_head_)

        if recognition_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            if train_cfg:
                recognition_head_train_cfg = train_cfg.recognition_head
            else:
                recognition_head_train_cfg = None
            recognition_head.update(train_cfg=recognition_head_train_cfg)
            recognition_head.update(test_cfg=test_cfg.recognition_head)
            self.recognition_head = build_head(recognition_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_detect_head(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'detect_head') and self.detect_head is not None

    @property
    def with_recognition_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(
            self, 'recognition_head') and self.recognition_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      use_gt=True,
                      **kwargs):
        x = self.extract_feat(img)

        losses = dict()

        preds = self.bbox_head(x)
        rpn_losses = self.bbox_head.loss(preds, **kwargs)
        losses.update(rpn_losses)

        if use_gt:
            proposal_list = [None for _ in range(len(img_metas))]
        else:
            proposal_list = self.bbox_head.get_boundary(preds)

        roi_losses = self.recognition_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks, **kwargs)
        losses.update(roi_losses)
        return losses

    def simple_test(self, img, img_metas, rescale=False):

        x = self.extract_feat(img)
        outs = self.detect_head(x)
        proposal_list = [
            self.detect_head.get_boundary(*(outs[i].unsqueeze(0)),
                                          [img_metas[i]], rescale)
            for i in range(len(img_metas))
        ]

        return self.recognition_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
