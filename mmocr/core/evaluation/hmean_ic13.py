# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

import mmocr.utils as utils
from . import utils as eval_utils


def compute_recall_precision(gt_polys, pred_polys):
    """Compute the recall and the precision matrices between gt and predicted
    polygons.

    Args:
        gt_polys (list[Polygon]): List of gt polygons.
        pred_polys (list[Polygon]): List of predicted polygons.

    Returns:
        recall (ndarray): Recall matrix of size gt_num x det_num.
        precision (ndarray): Precision matrix of size gt_num x det_num.
    """
    assert isinstance(gt_polys, list)
    assert isinstance(pred_polys, list)

    gt_num = len(gt_polys)
    det_num = len(pred_polys)
    sz = [gt_num, det_num]

    recall = np.zeros(sz)
    precision = np.zeros(sz)
    # compute area recall and precision for each (gt, det) pair
    # in one img
    for gt_id in range(gt_num):
        for pred_id in range(det_num):
            gt = gt_polys[gt_id]
            det = pred_polys[pred_id]

            inter_area, _ = eval_utils.poly_intersection(det, gt)
            gt_area = gt.area()
            det_area = det.area()
            if gt_area != 0:
                recall[gt_id, pred_id] = inter_area / gt_area
            if det_area != 0:
                precision[gt_id, pred_id] = inter_area / det_area

    return recall, precision


def eval_hmean_ic13(det_boxes,
                    gt_boxes,
                    gt_ignored_boxes,
                    precision_thr=0.4,
                    recall_thr=0.8,
                    center_dist_thr=1.0,
                    one2one_score=1.,
                    one2many_score=0.8,
                    many2one_score=1.):
    """Evalute hmean of text detection using the icdar2013 standard.

    Args:
        det_boxes (list[list[list[float]]]): List of arrays of shape (n, 2k).
            Each element is the det_boxes for one img. k>=4.
        gt_boxes (list[list[list[float]]]): List of arrays of shape (m, 2k).
            Each element is the gt_boxes for one img. k>=4.
        gt_ignored_boxes (list[list[list[float]]]): List of arrays of
            (l, 2k). Each element is the ignored gt_boxes for one img. k>=4.
        precision_thr (float): Precision threshold of the iou of one
            (gt_box, det_box) pair.
        recall_thr (float): Recall threshold of the iou of one
            (gt_box, det_box) pair.
        center_dist_thr (float): Distance threshold of one (gt_box, det_box)
            center point pair.
        one2one_score (float): Reward when one gt matches one det_box.
        one2many_score (float): Reward when one gt matches many det_boxes.
        many2one_score (float): Reward when many gts match one det_box.

    Returns:
        hmean (tuple[dict]): Tuple of dicts which encodes the hmean for
        the dataset and all images.
    """
    assert utils.is_3dlist(det_boxes)
    assert utils.is_3dlist(gt_boxes)
    assert utils.is_3dlist(gt_ignored_boxes)

    assert 0 <= precision_thr <= 1
    assert 0 <= recall_thr <= 1
    assert center_dist_thr > 0
    assert 0 <= one2one_score <= 1
    assert 0 <= one2many_score <= 1
    assert 0 <= many2one_score <= 1

    img_num = len(det_boxes)
    assert img_num == len(gt_boxes)
    assert img_num == len(gt_ignored_boxes)

    dataset_gt_num = 0
    dataset_pred_num = 0
    dataset_hit_recall = 0.0
    dataset_hit_prec = 0.0

    img_results = []

    for i in range(img_num):
        gt = gt_boxes[i]
        gt_ignored = gt_ignored_boxes[i]
        pred = det_boxes[i]

        gt_num = len(gt)
        ignored_num = len(gt_ignored)
        pred_num = len(pred)

        accum_recall = 0.
        accum_precision = 0.

        gt_points = gt + gt_ignored
        gt_polys = [eval_utils.points2polygon(p) for p in gt_points]
        gt_ignored_index = [gt_num + i for i in range(len(gt_ignored))]
        gt_num = len(gt_polys)

        pred_polys, pred_points, pred_ignored_index = eval_utils.ignore_pred(
            pred, gt_ignored_index, gt_polys, precision_thr)

        if pred_num > 0 and gt_num > 0:

            gt_hit = np.zeros(gt_num, np.int8).tolist()
            pred_hit = np.zeros(pred_num, np.int8).tolist()

            # compute area recall and precision for each (gt, pred) pair
            # in one img.
            recall_mat, precision_mat = compute_recall_precision(
                gt_polys, pred_polys)

            # match one gt to one pred box.
            for gt_id in range(gt_num):
                for pred_id in range(pred_num):
                    if (gt_hit[gt_id] != 0 or pred_hit[pred_id] != 0
                            or gt_id in gt_ignored_index
                            or pred_id in pred_ignored_index):
                        continue
                    match = eval_utils.one2one_match_ic13(
                        gt_id, pred_id, recall_mat, precision_mat, recall_thr,
                        precision_thr)

                    if match:
                        gt_point = np.array(gt_points[gt_id])
                        det_point = np.array(pred_points[pred_id])

                        norm_dist = eval_utils.box_center_distance(
                            det_point, gt_point)
                        norm_dist /= eval_utils.box_diag(
                            det_point) + eval_utils.box_diag(gt_point)
                        norm_dist *= 2.0

                        if norm_dist < center_dist_thr:
                            gt_hit[gt_id] = 1
                            pred_hit[pred_id] = 1
                            accum_recall += one2one_score
                            accum_precision += one2one_score

            # match one gt to many det boxes.
            for gt_id in range(gt_num):
                if gt_id in gt_ignored_index:
                    continue
                match, match_det_set = eval_utils.one2many_match_ic13(
                    gt_id, recall_mat, precision_mat, recall_thr,
                    precision_thr, gt_hit, pred_hit, pred_ignored_index)

                if match:
                    gt_hit[gt_id] = 1
                    accum_recall += one2many_score
                    accum_precision += one2many_score * len(match_det_set)
                    for pred_id in match_det_set:
                        pred_hit[pred_id] = 1

            # match many gt to one det box. One pair of (det,gt) are matched
            # successfully if their recall, precision, normalized distance
            # meet some thresholds.
            for pred_id in range(pred_num):
                if pred_id in pred_ignored_index:
                    continue

                match, match_gt_set = eval_utils.many2one_match_ic13(
                    pred_id, recall_mat, precision_mat, recall_thr,
                    precision_thr, gt_hit, pred_hit, gt_ignored_index)

                if match:
                    pred_hit[pred_id] = 1
                    accum_recall += many2one_score * len(match_gt_set)
                    accum_precision += many2one_score
                    for gt_id in match_gt_set:
                        gt_hit[gt_id] = 1

        gt_care_number = gt_num - ignored_num
        pred_care_number = pred_num - len(pred_ignored_index)

        r, p, h = eval_utils.compute_hmean(accum_recall, accum_precision,
                                           gt_care_number, pred_care_number)

        img_results.append({'recall': r, 'precision': p, 'hmean': h})

        dataset_gt_num += gt_care_number
        dataset_pred_num += pred_care_number
        dataset_hit_recall += accum_recall
        dataset_hit_prec += accum_precision

    total_r, total_p, total_h = eval_utils.compute_hmean(
        dataset_hit_recall, dataset_hit_prec, dataset_gt_num, dataset_pred_num)

    dataset_results = {
        'num_gts': dataset_gt_num,
        'num_dets': dataset_pred_num,
        'num_recall': dataset_hit_recall,
        'num_precision': dataset_hit_prec,
        'recall': total_r,
        'precision': total_p,
        'hmean': total_h
    }

    return dataset_results, img_results
