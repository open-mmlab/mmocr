# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from operator import itemgetter

import mmcv
import numpy as np
from mmcv.utils import print_log

import mmocr.utils as utils
from mmocr.core.evaluation import hmean_ic13, hmean_iou
from mmocr.core.evaluation.utils import (filter_2dlist_result,
                                         select_top_boundary)
from mmocr.core.mask import extract_boundary


def output_ranklist(img_results, img_infos, out_file):
    """Output the worst results for debugging.

    Args:
        img_results (list[dict]): Image result list.
        img_infos (list[dict]): Image information list.
        out_file (str): The output file path.

    Returns:
        sorted_results (list[dict]): Image results sorted by hmean.
    """
    assert utils.is_type_list(img_results, dict)
    assert utils.is_type_list(img_infos, dict)
    assert isinstance(out_file, str)
    assert out_file.endswith('json')

    sorted_results = []
    for idx, result in enumerate(img_results):
        name = img_infos[idx]['file_name']
        img_result = result
        img_result['file_name'] = name
        sorted_results.append(img_result)
    sorted_results = sorted(
        sorted_results, key=itemgetter('hmean'), reverse=False)

    mmcv.dump(sorted_results, file=out_file)

    return sorted_results


def get_gt_masks(ann_infos):
    """Get ground truth masks and ignored masks.

    Args:
        ann_infos (list[dict]): Each dict contains annotation
            infos of one image, containing following keys:
            masks, masks_ignore.
    Returns:
        gt_masks (list[list[list[int]]]): Ground truth masks.
        gt_masks_ignore (list[list[list[int]]]): Ignored masks.
    """
    assert utils.is_type_list(ann_infos, dict)

    gt_masks = []
    gt_masks_ignore = []
    for ann_info in ann_infos:
        masks = ann_info['masks']
        mask_gt = []
        for mask in masks:
            assert len(mask[0]) >= 8 and len(mask[0]) % 2 == 0
            mask_gt.append(mask[0])
        gt_masks.append(mask_gt)

        masks_ignore = ann_info['masks_ignore']
        mask_gt_ignore = []
        for mask_ignore in masks_ignore:
            assert len(mask_ignore[0]) >= 8 and len(mask_ignore[0]) % 2 == 0
            mask_gt_ignore.append(mask_ignore[0])
        gt_masks_ignore.append(mask_gt_ignore)

    return gt_masks, gt_masks_ignore


def eval_hmean(results,
               img_infos,
               ann_infos,
               metrics={'hmean-iou'},
               score_thr=None,
               min_score_thr=0.3,
               max_score_thr=0.9,
               step=0.1,
               rank_list=None,
               logger=None,
               **kwargs):
    """Evaluation in hmean metric. It conducts grid search over a range of
    boundary score thresholds and reports the best result.

    Args:
        results (list[dict]): Each dict corresponds to one image,
            containing the following keys: boundary_result
        img_infos (list[dict]): Each dict corresponds to one image,
            containing the following keys: filename, height, width
        ann_infos (list[dict]): Each dict corresponds to one image,
            containing the following keys: masks, masks_ignore
        score_thr (float): Deprecated. Please use min_score_thr instead.
        min_score_thr (float): Minimum score threshold of prediction map.
        max_score_thr (float): Maximum score threshold of prediction map.
        step (float): The spacing between score thresholds.
        metrics (set{str}): Hmean metric set, should be one or all of
            {'hmean-iou', 'hmean-ic13'}
    Returns:
        dict[str: float]
    """
    assert utils.is_type_list(results, dict)
    assert utils.is_type_list(img_infos, dict)
    assert utils.is_type_list(ann_infos, dict)

    if score_thr:
        warnings.warn('score_thr is deprecated. Please use min_score_thr '
                      'instead.')
        min_score_thr = score_thr

    assert 0 <= min_score_thr <= max_score_thr <= 1
    assert 0 <= step <= 1
    assert len(results) == len(img_infos) == len(ann_infos)
    assert isinstance(metrics, set)

    min_score_thr = float(min_score_thr)
    max_score_thr = float(max_score_thr)
    step = float(step)

    gts, gts_ignore = get_gt_masks(ann_infos)

    preds = []
    pred_scores = []
    for result in results:
        _, texts, scores = extract_boundary(result)
        if len(texts) > 0:
            assert utils.valid_boundary(texts[0], False)
        valid_texts, valid_text_scores = filter_2dlist_result(
            texts, scores, min_score_thr)
        preds.append(valid_texts)
        pred_scores.append(valid_text_scores)

    eval_results = {}

    for metric in metrics:
        msg = f'Evaluating {metric}...'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)
        best_result = dict(hmean=-1)
        for thr in np.arange(min_score_thr, min(max_score_thr + step, 1.0),
                             step):
            top_preds = select_top_boundary(preds, pred_scores, thr)
            if metric == 'hmean-iou':
                result, img_result = hmean_iou.eval_hmean_iou(
                    top_preds, gts, gts_ignore)
            elif metric == 'hmean-ic13':
                result, img_result = hmean_ic13.eval_hmean_ic13(
                    top_preds, gts, gts_ignore)
            else:
                raise NotImplementedError
            if rank_list is not None:
                output_ranklist(img_result, img_infos, rank_list)

            print_log(
                'thr {0:.2f}, recall: {1[recall]:.3f}, '
                'precision: {1[precision]:.3f}, '
                'hmean: {1[hmean]:.3f}'.format(thr, result),
                logger=logger)
            if result['hmean'] > best_result['hmean']:
                best_result = result
        eval_results[metric + ':recall'] = best_result['recall']
        eval_results[metric + ':precision'] = best_result['precision']
        eval_results[metric + ':hmean'] = best_result['hmean']
    return eval_results
