# Copyright (c) OpenMMLab. All rights reserved.
import torch
# from mmcv.runner import force_fp32
from mmcv.ops import batched_nms
from mmdet.core.anchor.point_generator import MlvlPointGenerator
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl

from .base_postprocessor import BaseTextDetPostProcessor


class ABCNetTextDetProcessor(BaseTextDetPostProcessor):

    def __init__(self, prior_generator, text_repr_type='poly'):
        super().__init__(text_repr_type=text_repr_type)
        self.prior_generator = MlvlPointGenerator()

    # @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def filter_and_location(self,
                            cls_scores,
                            bbox_preds,
                            score_factors=None,
                            img_metas=None,
                            cfg=None,
                            rescale=False,
                            with_nms=True,
                            **kwargs):

        assert len(cls_scores) == len(bbox_preds)
        assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            score_factor_list = select_single_mlvl(score_factors, img_id)

            results = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                              score_factor_list, mlvl_priors,
                                              img_meta, cfg, rescale, with_nms,
                                              **kwargs)
            result_list.append(results)
        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):

        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        mlvl_score_factors = []

        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(
                    bbox_pred=bbox_pred,
                    priors=priors,
                    score_factor=score_factor,
                ))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']
            score_factor = score_factor[keep_idxs]

            bboxes = self.bbox_coder.decode(
                priors, bbox_pred, max_shape=img_shape)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            mlvl_score_factors.append(score_factor)

        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes,
                                       img_meta['scale_factor'], cfg, rescale,
                                       with_nms, mlvl_score_factors, **kwargs)

    def _bbox_post_process(self,
                           mlvl_scores,
                           mlvl_labels,
                           mlvl_bboxes,
                           scale_factor,
                           cfg,
                           mlvl_score_factors=None,
                           **kwargs):

        assert len(mlvl_scores) == len(mlvl_bboxes) == len(mlvl_labels)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)

        if mlvl_score_factors is not None:
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors

        if mlvl_bboxes.numel() == 0:
            det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)
            return det_bboxes, mlvl_labels

        det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores,
                                            mlvl_labels, cfg.nms)
        det_bboxes = det_bboxes[:cfg.max_per_img]
        det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]
        return det_bboxes, det_labels
