# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import BaseModule
from mmdet.core import bbox2roi

from ..builder import (build_backbone, build_data_converter, build_head,
                       build_neck, build_postprocessor, build_recognizer,
                       build_roi_extractor)


class TwoStageTextSpotter(BaseModule):

    def __init__(self,
                 backbone=None,
                 neck=None,
                 bbox_head=None,
                 det_postprocess=None,
                 roi_extractor=None,
                 recognizer=None,
                 e2e_postprocess=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(TwoStageTextSpotter, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)

        if bbox_head is not None:
            bbox_head_train_cfg = None
            if train_cfg:
                bbox_head_train_cfg = train_cfg.bbox_head
            bbox_head_test_cfg = None
            if test_cfg:
                bbox_head_test_cfg = test_cfg.bbox_head
            bbox_head_ = bbox_head.copy()
            bbox_head_.update(
                train_cfg=bbox_head_train_cfg, test_cfg=bbox_head_test_cfg)
            self.bbox_head = build_head(bbox_head_)

        if det_postprocess:
            self.det_postprocess = build_postprocessor(det_postprocess)

        if roi_extractor is not None:
            roi_extractor_train_cfg = None
            if train_cfg:
                roi_extractor_train_cfg = train_cfg.roi_extractor
            roi_extractor_test_cfg = None
            if test_cfg:
                roi_extractor_test_cfg = test_cfg.roi_extractor
            roi_extractor_ = roi_extractor.copy()
            roi_extractor_.update(
                train_cfg=roi_extractor_train_cfg,
                test_cfg=roi_extractor_test_cfg)
            self.roi_extractor = build_roi_extractor(roi_extractor_)

        # convert the data format from detection to recognition
        if train_cfg:
            data_converter = train_cfg.data_converter
            self.data_converter = build_data_converter(data_converter)

        if recognizer is not None:
            self.recognizer = build_recognizer(recognizer)

        if e2e_postprocess:
            self.e2e_postprocess = build_postprocessor(e2e_postprocess)

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
                      **kwargs):
        loss = dict()
        x = self.extract_feat(img)
        preds = self.bbox_head(x)
        detect_losses = self.bbox_head.loss(preds, **kwargs)
        loss.update(detect_losses)
        proposal_list = self.postprocess(preds)
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(proposal_list[i],
                                                      gt_bboxes[i],
                                                      gt_bboxes_ignore[i],
                                                      gt_labels[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois, preds)
        recognize_loss = self.recognizer.forward_train(
            bbox_feats, self.data_converter(rois))
        loss.update(recognize_loss)
        return loss

    def simple_test(self, img, img_metas):
        x = self.extract_feat(img)
        preds = self.bbox_head(x)
        proposal_list = self.postprocess(preds)
        rois = bbox2roi(proposal_list)
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois, preds)
        recognizer_res = self.recognizer.simple_test(bbox_feats)
        output = self.e2e_postprocess(proposal_list, recognizer_res, img_metas)
        return output
