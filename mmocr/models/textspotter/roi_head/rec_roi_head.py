# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.core import bbox2result, bbox2roi
from mmdet.models import StandardRoIHead
from mmdet.models.builder import HEADS, build_roi_extractor
from mmdet.models.roi_heads.test_mixins import BBoxTestMixin, MaskTestMixin

from mmocr.models.builder import build_recognizer
from .text_mixins import RecTestMixin


@HEADS.register_module()
class RecRoIHead(StandardRoIHead, BBoxTestMixin, MaskTestMixin, RecTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 recognition_roi_extractor=None,
                 recognition_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        if recognition_roi_extractor is not None:
            self.init_recognition_roi_extractor_head(recognition_roi_extractor,
                                                     recognition_head)

    @property
    def with_recognition(self):
        """bool: whether the RoI head contains a `mask_head`"""
        return hasattr(
            self, 'recognition_head') and self.recognition_head is not None

    def init_recognition_head(self, recognition_roi_extractor,
                              recognition_head):
        """Initialize ``recognition_head``"""
        if recognition_roi_extractor is not None:
            self.recognition_roi_extractor = build_roi_extractor(
                recognition_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.recognition_roi_extractor = self.bbox_roi_extractor
        self.recognition_head = build_recognizer(recognition_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        # recognition head
        if self.with_recognition:
            recognition_rois = rois[:100]
            recognition_results = self._recognition_forward(
                x, recognition_rois)
            outs = outs + (recognition_results['recognition_pred'], )
        return outs

    def _recognition_forward_train(self, x, sampling_results, bbox_feats):
        """Run forward function and calculate loss for box head in training."""
        pos_rois, pos_inds = None, None
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)
        if pos_rois:
            recognition_feats = self.recognition_roi_extractor(
                x[:self.recognition_feats_roi_extractor.num_inputs], pos_rois)
            if self.with_shared_head:
                recognition_feats = self.shared_head(recognition_feats)
        else:
            assert bbox_feats is not None
            recognition_feats = bbox_feats[pos_inds]

        loss_recognition = self.recognition_head.forward_train(
            recognition_feats, sampling_results.gt_text)

        recognition_results = dict(loss_recognition=loss_recognition)
        return recognition_results

    def _recognition_forward(self):
        pass

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        assert self.with_recognition, 'recognition head must be implemented.'
        args = dict()
        if self.with_bbox:
            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

            bbox_results = [
                bbox2result(det_bboxes[i], det_labels[i],
                            self.bbox_head.num_classes)
                for i in range(len(det_bboxes))
            ]
            args.update(
                det_bboxes=det_bboxes,
                det_labels=det_labels,
                bbox_results=bbox_results)
        if self.with_mask:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            args.update(segm_results=segm_results)
        recognition_results = self.simple_test_recognition(
            x, img_metas, proposal_list, rescale, **args)
        return recognition_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        # mask head forward and loss
        if self.with_recognition:
            if locals().get('bbox_results', None):
                bbox_feats = bbox_results['bbox_feats']
            else:
                bbox_feats = None
            recognition_results = self._recognition_forward_train(
                x, sampling_results, bbox_feats)

            losses.update(recognition_results['loss_recognition'])

        return losses
