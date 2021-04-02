import numpy as np
import Polygon as plg

import mmocr.utils as utils


def ignore_pred(pred_boxes, gt_ignored_index, gt_polys, precision_thr):
    """Ignore the predicted box if it hits any ignored ground truth.

    Args:
        pred_boxes (list[ndarray or list]): The predicted boxes of one image.
        gt_ignored_index (list[int]): The ignored ground truth index list.
        gt_polys (list[Polygon]): The polygon list of one image.
        precision_thr (float): The precision threshold.

    Returns:
        pred_polys (list[Polygon]): The predicted polygon list.
        pred_points (list[list]): The predicted box list represented
            by point sequences.
        pred_ignored_index (list[int]): The ignored text index list.
    """

    assert isinstance(pred_boxes, list)
    assert isinstance(gt_ignored_index, list)
    assert isinstance(gt_polys, list)
    assert 0 <= precision_thr <= 1

    pred_polys = []
    pred_points = []
    pred_ignored_index = []

    gt_dont_care_num = len(gt_ignored_index)
    # get detection polygons
    for box_id, box in enumerate(pred_boxes):
        poly = points2polygon(box)
        pred_polys.append(poly)
        pred_points.append(box)

        if gt_dont_care_num < 1:
            continue

        # ignore the current detection box
        # if its overlap with any ignored gt > precision_thr
        for dont_care_box_id in gt_ignored_index:
            dont_care_box = gt_polys[dont_care_box_id]
            inter_area, _ = poly_intersection(poly, dont_care_box)
            area = poly.area()
            precision = 0 if area == 0 \
                else inter_area / area
            if precision > precision_thr:
                pred_ignored_index.append(box_id)
                break

    return pred_polys, pred_points, pred_ignored_index


def compute_hmean(accum_hit_recall, accum_hit_prec, gt_num, pred_num):
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


def box2polygon(box):
    """Convert box to polygon.

    Args:
        box (ndarray or list): A ndarray or a list of shape (4)
            that indicates 2 points.

    Returns:
        polygon (Polygon): A polygon object.
    """
    if isinstance(box, list):
        box = np.array(box)

    assert isinstance(box, np.ndarray)
    assert box.size == 4
    boundary = np.array(
        [box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3]])

    point_mat = boundary.reshape([-1, 2])
    return plg.Polygon(point_mat)


def points2polygon(points):
    """Convert k points to 1 polygon.

    Args:
        points (ndarray or list): A ndarray or a list of shape (2k)
            that indicates k points.

    Returns:
        polygon (Polygon): A polygon object.
    """
    if isinstance(points, list):
        points = np.array(points)

    assert isinstance(points, np.ndarray)
    assert (points.size % 2 == 0) and (points.size >= 8)

    point_mat = points.reshape([-1, 2])
    return plg.Polygon(point_mat)


def poly_intersection(poly_det, poly_gt):
    """Calculate the intersection area between two polygon.

    Args:
        poly_det (Polygon): A polygon predicted by detector.
        poly_gt (Polygon): A gt polygon.

    Returns:
        intersection_area (float): The intersection area between two polygons.
    """
    assert isinstance(poly_det, plg.Polygon)
    assert isinstance(poly_gt, plg.Polygon)

    poly_inter = poly_det & poly_gt
    if len(poly_inter) == 0:
        return 0, poly_inter
    return poly_inter.area(), poly_inter


def poly_union(poly_det, poly_gt):
    """Calculate the union area between two polygon.

    Args:
        poly_det (Polygon): A polygon predicted by detector.
        poly_gt (Polygon): A gt polygon.

    Returns:
        union_area (float): The union area between two polygons.
    """
    assert isinstance(poly_det, plg.Polygon)
    assert isinstance(poly_gt, plg.Polygon)

    area_det = poly_det.area()
    area_gt = poly_gt.area()
    area_inters, _ = poly_intersection(poly_det, poly_gt)
    return area_det + area_gt - area_inters


def boundary_iou(src, target):
    """Calculate the IOU between two boundaries.

    Args:
       src (list): Source boundary.
       target (list): Target boundary.

    Returns:
       iou (float): The iou between two boundaries.
    """
    assert utils.valid_boundary(src, False)
    assert utils.valid_boundary(target, False)
    src_poly = points2polygon(src)
    target_poly = points2polygon(target)

    return poly_iou(src_poly, target_poly)


def poly_iou(poly_det, poly_gt):
    """Calculate the IOU between two polygons.

    Args:
        poly_det (Polygon): A polygon predicted by detector.
        poly_gt (Polygon): A gt polygon.

    Returns:
        iou (float): The IOU between two polygons.
    """
    assert isinstance(poly_det, plg.Polygon)
    assert isinstance(poly_gt, plg.Polygon)
    area_inters, _ = poly_intersection(poly_det, poly_gt)

    return area_inters / poly_union(poly_det, poly_gt)


def one2one_match_ic13(gt_id, det_id, recall_mat, precision_mat, recall_thr,
                       precision_thr):
    """One-to-One match gt and det with icdar2013 standards.

    Args:
        gt_id (int): The ground truth id index.
        det_id (int): The detection result id index.
        recall_mat (ndarray): gt_numxdet_num matrix with element (i,j) being
            the recall ratio of gt i to det j.
        precision_mat (ndarray): gt_numxdet_num matrix with element (i,j) being
            the precision ratio of gt i to det j.
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


def one2many_match_ic13(gt_id, recall_mat, precision_mat, recall_thr,
                        precision_thr, gt_match_flag, det_match_flag,
                        det_dont_care_index):
    """One-to-Many match gt and detections with icdar2013 standards.

    Args:
        gt_id (int): gt index.
        recall_mat (ndarray): gt_numxdet_num matrix with element (i,j) being
            the recall ratio of gt i to det j.
        precision_mat (ndarray): gt_numxdet_num matrix with element (i,j) being
            the precision ratio of gt i to det j.
        recall_thr (float): The recall threshold.
        precision_thr (float): The precision threshold.
        gt_match_flag (ndarray): An array indicates each gt matched already.
        det_match_flag (ndarray): An array indicates each box has been
            matched already or not.
        det_dont_care_index (list): A list indicates each detection box can be
            ignored or not.

    Returns:
        tuple (True|False, list): The first indicates the gt is matched or not;
            the second is the matched detection ids.
    """
    assert isinstance(gt_id, int)
    assert isinstance(recall_mat, np.ndarray)
    assert isinstance(precision_mat, np.ndarray)
    assert 0 <= recall_thr <= 1
    assert 0 <= precision_thr <= 1

    assert isinstance(gt_match_flag, list)
    assert isinstance(det_match_flag, list)
    assert isinstance(det_dont_care_index, list)

    many_sum = 0.
    det_ids = []
    for det_id in range(recall_mat.shape[1]):
        if gt_match_flag[gt_id] == 0 and det_match_flag[
                det_id] == 0 and det_id not in det_dont_care_index:
            if precision_mat[gt_id, det_id] >= precision_thr:
                many_sum += recall_mat[gt_id, det_id]
                det_ids.append(det_id)
    if many_sum >= recall_thr:
        return True, det_ids
    return False, []


def many2one_match_ic13(det_id, recall_mat, precision_mat, recall_thr,
                        precision_thr, gt_match_flag, det_match_flag,
                        gt_dont_care_index):
    """Many-to-One match gt and detections with icdar2013 standards.

    Args:
        det_id (int): Detection index.
        recall_mat (ndarray): gt_numxdet_num matrix with element (i,j) being
            the recall ratio of gt i to det j.
        precision_mat (ndarray): gt_numxdet_num matrix with element (i,j) being
            the precision ratio of gt i to det j.
        recall_thr (float): The recall threshold.
        precision_thr (float): The precision threshold.
        gt_match_flag (ndarray): An array indicates each gt has been matched
            already.
        det_match_flag (ndarray): An array indicates each detection box has
            been matched already or not.
        gt_dont_care_index (list): A list indicates each gt box can be ignored
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
    assert isinstance(gt_dont_care_index, list)
    many_sum = 0.
    gt_ids = []
    for gt_id in range(recall_mat.shape[0]):
        if gt_match_flag[gt_id] == 0 and det_match_flag[
                det_id] == 0 and gt_id not in gt_dont_care_index:
            if recall_mat[gt_id, det_id] >= recall_thr:
                many_sum += precision_mat[gt_id, det_id]
                gt_ids.append(gt_id)
    if many_sum >= precision_thr:
        return True, gt_ids
    return False, []


def points_center(points):

    assert isinstance(points, np.ndarray)
    assert points.size % 2 == 0

    points = points.reshape([-1, 2])
    return np.mean(points, axis=0)


def point_distance(p1, p2):
    assert isinstance(p1, np.ndarray)
    assert isinstance(p2, np.ndarray)

    assert p1.size == 2
    assert p2.size == 2

    dist = np.square(p2 - p1)
    dist = np.sum(dist)
    dist = np.sqrt(dist)
    return dist


def box_center_distance(b1, b2):
    assert isinstance(b1, np.ndarray)
    assert isinstance(b2, np.ndarray)
    return point_distance(points_center(b1), points_center(b2))


def box_diag(box):
    assert isinstance(box, np.ndarray)
    assert box.size == 8

    return point_distance(box[0:2], box[4:6])


def filter_2dlist_result(results, scores, score_thr):
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
    valid_results = [results[inx] for inx in np.where(inds)[0].tolist()]
    valid_scores = [scores[inx] for inx in np.where(inds)[0].tolist()]
    return valid_results, valid_scores


def filter_result(results, scores, score_thr):
    """Find out detected results whose score > score_thr.

    Args:
        results (ndarray): The results matrix of shape (n, k).
        score (ndarray): The score vector of shape (n,).
        score_thr (float): The score threshold.
    Returns:
        valid_results (ndarray): The valid results of shape (m,k) with m<=n.
        valid_score (ndarray): The scores which correspond to the
            valid results.
    """
    assert results.ndim == 2
    assert scores.shape[0] == results.shape[0]
    assert isinstance(score_thr, float)
    assert 0 <= score_thr <= 1

    inds = scores > score_thr
    valid_results = results[inds, :]
    valid_scores = scores[inds]
    return valid_results, valid_scores


def select_top_boundary(boundaries_list, scores_list, score_thr):
    """Select poly boundaries with scores >= score_thr.

    Args:
        boundaries_list (list[list[list[float]]]): List of boundaries.
            The 1st, 2rd, and 3rd indices are for image, text and
            vertice, respectively.
        scores_list (list(list[float])): List of lists of scores.
        score_thr (float): The score threshold to filter out bboxes.

    Returns:
        selected_bboxes (list[list[list[float]]]): List of boundaries.
            The 1st, 2rd, and 3rd indices are for image, text and vertice,
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


def select_bboxes_via_score(bboxes_list, scores_list, score_thr):
    """Select bboxes with scores >= score_thr.

    Args:
        bboxes_list (list[ndarray]): List of bboxes. Each element is ndarray of
            shape (n,8)
        scores_list (list(list[float])): List of lists of scores.
        score_thr (float): The score threshold to filter out bboxes.

    Returns:
        selected_bboxes (list[ndarray]): List of bboxes. Each element is
            ndarray of shape (m,8) with m<=n.
    """
    assert isinstance(bboxes_list, list)
    assert isinstance(scores_list, list)
    assert isinstance(score_thr, float)
    assert len(bboxes_list) == len(scores_list)
    assert 0 <= score_thr <= 1

    selected_bboxes = []
    for bboxes, scores in zip(bboxes_list, scores_list):
        if len(scores) > 0:
            assert len(scores) == bboxes.shape[0]
            inds = [
                iter for iter in range(len(scores))
                if scores[iter] >= score_thr
            ]
            selected_bboxes.append(bboxes[inds, :])
        else:
            selected_bboxes.append(bboxes)
    return selected_bboxes
