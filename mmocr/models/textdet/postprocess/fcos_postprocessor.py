# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
import torch
# from mmcv.runner import force_fp32
from mmcv.ops import batched_nms
from mmdet.core import build_bbox_coder, multi_apply
from mmdet.core.anchor.point_generator import MlvlPointGenerator
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl

from mmocr.models.builder import POSTPROCESSOR
from mmocr.utils.box_util import bbox_to_polygon, bezier_to_polygon
from .base_postprocessor import BaseTextDetPostProcessor


@POSTPROCESSOR.register_module()
class FCOSPostprocessor(BaseTextDetPostProcessor):

    def __init__(self,
                 num_classes=1,
                 use_sigmoid_cls=True,
                 strides=(4, 8, 16, 32, 64),
                 norm_by_strides=True,
                 bbox_coder=dict(type='DistancePointBBoxCoder'),
                 text_repr_type='poly',
                 with_bezier=False,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super().__init__(
            text_repr_type=text_repr_type,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs)
        self.strides = strides
        self.norm_by_strides = norm_by_strides
        self.prior_generator = MlvlPointGenerator(strides)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.use_sigmoid_cls = use_sigmoid_cls
        self.with_bezier = with_bezier
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

    # @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_text_instance(
            self,
            pred_result,
            img_metas,
            filter_and_location=True,
            reconstruct=True,
            nms_pre=-1,
            score_thr=0,
            max_per_img=100,
            nms=dict(type='nms', iou_threshold=0.5),
    ):
        if filter_and_location:
            results = self.filter_and_location(
                pred_result,
                img_metas,
                nms_pre,
                score_thr,
                max_per_img,
                nms,
            )
        else:
            results = copy.deepcopy(pred_result)

        if reconstruct:
            results = self.reconstruct_text_instance(results)
        return results

    def filter_and_location(self,
                            det_results,
                            img_metas=None,
                            nms_pre=-1,
                            score_thr=0,
                            max_per_img=100,
                            nms=dict(type='nms', iou_threshold=0.5)):
        cls_scores = det_results.get('cls_scores')
        centernesses = det_results.get('centernesses')
        bbox_preds = det_results.get('bbox_preds')
        num_levels = len(cls_scores)
        bezier_preds = det_results.get('bezier_preds',
                                       [None for _ in range(num_levels)])
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)
        parameters = dict(
            img_shape=img_metas['img_shape'],
            nms_pre=nms_pre,
            score_thr=score_thr)

        (mlvl_bboxes, mlvl_scores, mlvl_labels, mlvl_score_factors,
         mlvl_beziers) = multi_apply(self._single_level, cls_scores,
                                     bbox_preds, centernesses, bezier_preds,
                                     mlvl_priors, self.strides, **parameters)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)
        if self.with_bezier:
            mlvl_beziers = torch.cat(mlvl_beziers)

        if mlvl_score_factors is not None:
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors

        if mlvl_bboxes.numel() == 0:
            results = dict(
                bboxes=mlvl_bboxes.detach().cpu().numpy(),
                scores=mlvl_scores[:, None].detach().cpu().numpy(),
                labels=mlvl_labels.detach().cpu().numpy())
            if self.with_bezier:
                results['beziers'] = mlvl_beziers.detach().cpu().numpy()
            return results
        det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores,
                                            mlvl_labels, nms)
        det_bboxes, scores = np.split(det_bboxes, [-1], axis=1)
        results = dict(
            bboxes=det_bboxes[:max_per_img].detach().cpu().numpy(),
            scores=scores[:max_per_img].detach().cpu().numpy(),
            labels=mlvl_labels[keep_idxs][:max_per_img].detach().cpu().numpy())
        if self.with_bezier:
            results['beziers'] = mlvl_beziers[keep_idxs][:max_per_img].detach(
            ).cpu().numpy()

        return results

    def split_results(self, pred_results, img_metas, **kwargs):

        results = []
        cls_scores = pred_results.get('cls_scores')
        bbox_preds = pred_results.get('bbox_preds')
        centernesses = pred_results.get('centernesses')
        if self.with_bezier:
            bezier_preds = pred_results.get('bezier_preds')
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)
        for img_id in range(len(img_metas)):
            single_results = dict(
                cls_scores=select_single_mlvl(cls_scores, img_id),
                bbox_preds=select_single_mlvl(bbox_preds, img_id),
                centernesses=select_single_mlvl(centernesses, img_id),
                mlvl_priors=mlvl_priors)
            if self.with_bezier:
                single_results['bezier_preds'] = select_single_mlvl(
                    bezier_preds, img_id)
            results.append(single_results)
        return results

    def reconstruct_text_instance(self, results):
        if self.with_bezier:
            bezier_points = results['bezier_preds'].reshape(-1, 2, 4, 2)
            results['polygons'] = list(map(bezier_to_polygon, bezier_points))
        else:
            results['polygons'] = list(map(bbox_to_polygon, results['bboxes']))
        return results

    def _single_level(self, cls_scores, bbox_preds, centernesses, bezier_preds,
                      priors, stride, score_thr, nms_pre, img_shape):
        assert cls_scores.size()[-2:] == bbox_preds.size()[-2:]
        '''
        if self.norm_by_strides:
            bbox_pred = bbox_pred * stride
            bezier_pred = bezier_pred * stride
        '''
        bbox_preds = bbox_preds.permute(1, 2, 0).reshape(-1, 4)
        if self.with_bezier:
            bezier_preds = bezier_preds.permute(1, 2, 0).reshape(-1, 8, 2)
        centernesses = centernesses.permute(1, 2, 0).reshape(-1).sigmoid()
        cls_scores = cls_scores.permute(1, 2,
                                        0).reshape(-1, self.cls_out_channels)
        if self.use_sigmoid_cls:
            scores = cls_scores.sigmoid()
        else:
            # remind that we set FG labels to [0, num_class-1]
            # since mmdet v2.0
            # BG cat_id: num_class
            scores = cls_scores.softmax(-1)[:, :-1]

        # After https://github.com/open-mmlab/mmdetection/pull/6268/,
        # this operation keeps fewer bboxes under the same `nms_pre`.
        # There is no difference in performance for most models. If you
        # find a slight drop in performance, you can set a larger
        # `nms_pre` than before.
        results = filter_scores_and_topk(
            scores, score_thr, nms_pre,
            dict(bbox_preds=bbox_preds, priors=priors))
        scores, labels, keep_idxs, filtered_results = results

        bbox_preds = filtered_results['bbox_preds']
        priors = filtered_results['priors']
        centernesses = centernesses[keep_idxs]
        if self.with_bezier:
            bezier_preds = bezier_preds[keep_idxs]

        bboxes = self.bbox_coder.decode(
            priors, bbox_preds, max_shape=img_shape)
        if self.with_bezier:
            bezier_preds = priors[:, None, :] + bezier_preds
            bezier_preds[:, :, :0].clamp_(min=0, max=img_shape[1])
            bezier_preds[:, :, :1].clamp_(min=0, max=img_shape[0])
        return bboxes, scores, labels, centernesses, bezier_preds
