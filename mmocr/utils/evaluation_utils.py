# Copyright (c) OpenMMLab. All rights reserved.
# TODO check whether to keep these utils after refactoring ic13 metrics
import numpy as np


def compute_hmean(accum_hit_recall, accum_hit_prec, gt_num, pred_num):
    # TODO Add typehints & Test
    """Compute hmean given hit number, ground truth number and prediction
    number.

    Args:
        accum_hit_recall (int|float): Accumulated hits for computing recall.
        accum_hit_prec (int|float): Accumulated hits for computing precision.
        gt_num (int): Ground truth number.
        pred_num (int): Prediction number.

    Returns:
        recall (float):  The recall value.
        precision (float): The precision value.
        hmean (float): The hmean value.
    """

    assert isinstance(accum_hit_recall, (float, int))
    assert isinstance(accum_hit_prec, (float, int))

    assert isinstance(gt_num, int)
    assert isinstance(pred_num, int)
    assert accum_hit_recall >= 0.0
    assert accum_hit_prec >= 0.0
    assert gt_num >= 0.0
    assert pred_num >= 0.0

    if gt_num == 0:
        recall = 1.0
        precision = 0.0 if pred_num > 0 else 1.0
    else:
        recall = float(accum_hit_recall) / gt_num
        precision = 0.0 if pred_num == 0 else float(accum_hit_prec) / pred_num

    denom = recall + precision

    hmean = 0.0 if denom == 0 else (2.0 * precision * recall / denom)

    return recall, precision, hmean


def one2one_match_ic13(gt_id, det_id, recall_mat, precision_mat, recall_thr,
                       precision_thr):
    # TODO Add typehints & Test
    """One-to-One match gt and det with icdar2013 standards.

    Args:
        gt_id (int): The ground truth id index.
        det_id (int): The detection result id index.
        recall_mat (ndarray): `gt_num x det_num` matrix with element (i,j)
            being the recall ratio of gt i to det j.
        precision_mat (ndarray): `gt_num x det_num` matrix with element (i,j)
            being the precision ratio of gt i to det j.
        recall_thr (float): The recall threshold.
        precision_thr (float): The precision threshold.
    Returns:
        True|False: Whether the gt and det are matched.
    """
    assert isinstance(gt_id, int)
    assert isinstance(det_id, int)
    assert isinstance(recall_mat, np.ndarray)
    assert isinstance(precision_mat, np.ndarray)
    assert 0 <= recall_thr <= 1
    assert 0 <= precision_thr <= 1

    cont = 0
    for i in range(recall_mat.shape[1]):
        if recall_mat[gt_id,
                      i] > recall_thr and precision_mat[gt_id,
                                                        i] > precision_thr:
            cont += 1
    if cont != 1:
        return False

    cont = 0
    for i in range(recall_mat.shape[0]):
        if recall_mat[i, det_id] > recall_thr and precision_mat[
                i, det_id] > precision_thr:
            cont += 1
    if cont != 1:
        return False

    if recall_mat[gt_id, det_id] > recall_thr and precision_mat[
            gt_id, det_id] > precision_thr:
        return True

    return False


def many2one_match_ic13(det_id, recall_mat, precision_mat, recall_thr,
                        precision_thr, gt_match_flag, det_match_flag,
                        gt_ignored_index):
    # TODO Add typehints & Test
    """Many-to-One match gt and detections with icdar2013 standards.

    Args:
        det_id (int): Detection index.
        recall_mat (ndarray): `gt_num x det_num` matrix with element (i,j)
            being the recall ratio of gt i to det j.
        precision_mat (ndarray): `gt_num x det_num` matrix with element (i,j)
            being the precision ratio of gt i to det j.
        recall_thr (float): The recall threshold.
        precision_thr (float): The precision threshold.
        gt_match_flag (ndarray): An array indicates each gt has been matched
            already.
        det_match_flag (ndarray): An array indicates each detection box has
            been matched already or not.
        gt_ignored_index (list): A list indicates each gt box can be ignored
            or not.

    Returns:
        tuple (True|False, list): The first indicates the detection is matched
            or not; the second is the matched gt ids.
    """
    assert isinstance(det_id, int)
    assert isinstance(recall_mat, np.ndarray)
    assert isinstance(precision_mat, np.ndarray)
    assert 0 <= recall_thr <= 1
    assert 0 <= precision_thr <= 1

    assert isinstance(gt_match_flag, list)
    assert isinstance(det_match_flag, list)
    assert isinstance(gt_ignored_index, list)
    many_sum = 0.
    gt_ids = []
    for gt_id in range(recall_mat.shape[0]):
        if gt_match_flag[gt_id] == 0 and det_match_flag[
                det_id] == 0 and gt_id not in gt_ignored_index:
            if recall_mat[gt_id, det_id] >= recall_thr:
                many_sum += precision_mat[gt_id, det_id]
                gt_ids.append(gt_id)
    if many_sum >= precision_thr:
        return True, gt_ids
    return False, []


def filter_2dlist_result(results, scores, score_thr):
    # TODO Add typehints & Test
    """Find out detected results whose score > score_thr.

    Args:
        results (list[list[float]]): The result list.
        score (list): The score list.
        score_thr (float): The score threshold.
    Returns:
        valid_results (list[list[float]]): The valid results.
        valid_score (list[float]): The scores which correspond to the valid
            results.
    """
    assert isinstance(results, list)
    assert len(results) == len(scores)
    assert isinstance(score_thr, float)
    assert 0 <= score_thr <= 1

    inds = np.array(scores) > score_thr
    valid_results = [results[idx] for idx in np.where(inds)[0].tolist()]
    valid_scores = [scores[idx] for idx in np.where(inds)[0].tolist()]
    return valid_results, valid_scores


def select_top_boundary(boundaries_list, scores_list, score_thr):
    # TODO Add typehints & Test
    """Select poly boundaries with scores >= score_thr.

    Args:
        boundaries_list (list[list[list[float]]]): List of boundaries.
            The 1st, 2nd, and 3rd indices are for image, text and
            vertice, respectively.
        scores_list (list(list[float])): List of lists of scores.
        score_thr (float): The score threshold to filter out bboxes.

    Returns:
        selected_bboxes (list[list[list[float]]]): List of boundaries.
            The 1st, 2nd, and 3rd indices are for image, text and vertice,
            respectively.
    """
    assert isinstance(boundaries_list, list)
    assert isinstance(scores_list, list)
    assert isinstance(score_thr, float)
    assert len(boundaries_list) == len(scores_list)
    assert 0 <= score_thr <= 1

    selected_boundaries = []
    for boundary, scores in zip(boundaries_list, scores_list):
        if len(scores) > 0:
            assert len(scores) == len(boundary)
            inds = [
                iter for iter in range(len(scores))
                if scores[iter] >= score_thr
            ]
            selected_boundaries.append([boundary[i] for i in inds])
        else:
            selected_boundaries.append(boundary)
    return selected_boundaries
