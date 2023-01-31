# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import List

import numpy as np
import torch
from mmcv.ops import batched_nms
from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
from mmdet.models.utils import (filter_scores_and_topk, multi_apply,
                                select_single_mlvl)

from mmengine.structures import InstanceData
from mmocr.models.textdet.postprocessors.base import BaseTextDetPostProcessor
from mmocr.registry import MODELS, TASK_UTILS


@MODELS.register_module()
class ABCNetDetPostprocessor(BaseTextDetPostProcessor):
    """Post-processing methods for ABCNet.

    Args:
        num_classes (int): Number of classes.
        use_sigmoid_cls (bool): Whether to use sigmoid for classification.
        strides (tuple): Strides of each feature map.
        norm_by_strides (bool): Whether to normalize the regression targets by
            the strides.
        bbox_coder (dict): Config dict for bbox coder.
        text_repr_type (str): Text representation type, 'poly' or 'quad'.
        with_bezier (bool): Whether to use bezier curve for text detection.
        train_cfg (dict): Config dict for training.
        test_cfg (dict): Config dict for testing.
    """

    def __init__(
        self,
        num_classes=1,
        use_sigmoid_cls=True,
        strides=(4, 8, 16, 32, 64),
        norm_by_strides=True,
        bbox_coder=dict(type='mmdet.DistancePointBBoxCoder'),
        text_repr_type='poly',
        rescale_fields=None,
        with_bezier=False,
        train_cfg=None,
        test_cfg=None,
    ):
        super().__init__(
            text_repr_type=text_repr_type,
            rescale_fields=rescale_fields,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        self.strides = strides
        self.norm_by_strides = norm_by_strides
        self.prior_generator = MlvlPointGenerator(strides)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.use_sigmoid_cls = use_sigmoid_cls
        self.with_bezier = with_bezier
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

    def split_results(self, pred_results: List[torch.Tensor]):
        """Split the prediction results into multi-level features. The
        prediction results are concatenated in the first dimension.
        Args:
            pred_results (list[list[torch.Tensor]): Prediction results of all
                head with multi-level features.
                The first dimension of pred_results is the number of outputs of
                head. The second dimension is the number of level. The third
                dimension is the feature with (N, C, H, W).

        Returns:
            list[list[torch.Tensor]]:
            [Batch_size, Number of heads]
        """

        results = []
        num_levels = len(pred_results[0])
        bs = pred_results[0][0].size(0)
        featmap_sizes = [
            pred_results[0][i].shape[-2:] for i in range(num_levels)
        ]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=pred_results[0][0].dtype,
            device=pred_results[0][0].device)
        for img_id in range(bs):
            single_results = [mlvl_priors]
            for pred_result in pred_results:
                single_results.append(select_single_mlvl(pred_result, img_id))
            results.append(single_results)
        return results

    def get_text_instances(
            self,
            pred_results,
            data_sample,
            nms_pre=-1,
            score_thr=0,
            max_per_img=100,
            nms=dict(type='nms', iou_threshold=0.5),
    ):
        """Get text instance predictions of one image."""
        pred_instances = InstanceData()

        (mlvl_bboxes, mlvl_scores, mlvl_labels, mlvl_score_factors,
         mlvl_beziers) = multi_apply(
             self._get_preds_single_level,
             *pred_results,
             self.strides,
             img_shape=data_sample.get('img_shape'),
             nms_pre=nms_pre,
             score_thr=score_thr)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)
        if self.with_bezier:
            mlvl_beziers = torch.cat(mlvl_beziers)

        if mlvl_score_factors is not None:
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors
            mlvl_scores = torch.sqrt(mlvl_scores)

        if mlvl_bboxes.numel() == 0:
            pred_instances.bboxes = mlvl_bboxes.detach().cpu().numpy()
            pred_instances.scores = mlvl_scores.detach().cpu().numpy()
            pred_instances.labels = mlvl_labels.detach().cpu().numpy()
            if self.with_bezier:
                pred_instances.beziers = mlvl_beziers.detach().reshape(-1, 16)
            pred_instances.polygons = []
            data_sample.pred_instances = pred_instances
            return data_sample
        det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores,
                                            mlvl_labels, nms)
        det_bboxes, scores = np.split(det_bboxes, [-1], axis=1)
        pred_instances.bboxes = det_bboxes[:max_per_img].detach().cpu().numpy()
        pred_instances.scores = scores[:max_per_img].detach().cpu().numpy(
        ).squeeze(-1)
        pred_instances.labels = mlvl_labels[keep_idxs][:max_per_img].detach(
        ).cpu().numpy()
        if self.with_bezier:
            pred_instances.beziers = mlvl_beziers[
                keep_idxs][:max_per_img].detach().reshape(-1, 16)
        data_sample.pred_instances = pred_instances
        return data_sample

    def _get_preds_single_level(self,
                                priors,
                                cls_scores,
                                bbox_preds,
                                centernesses,
                                bezier_preds=None,
                                stride=1,
                                score_thr=0,
                                nms_pre=-1,
                                img_shape=None):
        assert cls_scores.size()[-2:] == bbox_preds.size()[-2:]
        if self.norm_by_strides:
            bbox_preds = bbox_preds * stride
        bbox_preds = bbox_preds.permute(1, 2, 0).reshape(-1, 4)
        if self.with_bezier:
            if self.norm_by_strides:
                bezier_preds = bezier_preds * stride
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
        bboxes = self.bbox_coder.decode(
            priors, bbox_preds, max_shape=img_shape)
        if self.with_bezier:
            bezier_preds = bezier_preds[keep_idxs]
            bezier_preds = priors[:, None, :] + bezier_preds
            bezier_preds[:, :, 0].clamp_(min=0, max=img_shape[1])
            bezier_preds[:, :, 1].clamp_(min=0, max=img_shape[0])
            return bboxes, scores, labels, centernesses, bezier_preds
        else:
            return bboxes, scores, labels, centernesses

    def __call__(self, pred_results, data_samples, training: bool = False):
        """Postprocess pred_results according to metainfos in data_samples.

        Args:
            pred_results (Union[Tensor, List[Tensor]]): The prediction results
                stored in a tensor or a list of tensor. Usually each item to
                be post-processed is expected to be a batched tensor.
            data_samples (list[TextDetDataSample]): Batch of data_samples,
                each corresponding to a prediction result.
            training (bool): Whether the model is in training mode. Defaults to
                False.

        Returns:
            list[TextDetDataSample]: Batch of post-processed datasamples.
        """
        if training:
            return data_samples
        cfg = self.train_cfg if training else self.test_cfg
        if cfg is None:
            cfg = {}
        pred_results = self.split_results(pred_results)
        process_single = partial(self._process_single, **cfg)
        results = list(map(process_single, pred_results, data_samples))

        return results
