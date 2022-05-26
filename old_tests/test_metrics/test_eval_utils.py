# Copyright (c) OpenMMLab. All rights reserved.
"""Tests the utils of evaluation."""
import numpy as np
import pytest

import mmocr.core.evaluation.utils as utils


def test_compute_hmean():

    # test invalid arguments
    with pytest.raises(AssertionError):
        utils.compute_hmean(0, 0, 0.0, 0)
    with pytest.raises(AssertionError):
        utils.compute_hmean(0, 0, 0, 0.0)
    with pytest.raises(AssertionError):
        utils.compute_hmean([1], 0, 0, 0)
    with pytest.raises(AssertionError):
        utils.compute_hmean(0, [1], 0, 0)

    _, _, hmean = utils.compute_hmean(2, 2, 2, 2)
    assert hmean == 1

    _, _, hmean = utils.compute_hmean(0, 0, 2, 2)
    assert hmean == 0


def test_boundary_iou():
    points = [0, 0, 0, 1, 1, 1, 1, 0]
    points1 = [10, 20, 30, 40, 50, 60, 70, 80]
    points2 = [0, 0, 0, 0, 0, 0, 0, 0]  # Invalid polygon
    points3 = [0, 0, 0, 1, 1, 0, 1, 1]  # Self-intersected polygon

    assert utils.boundary_iou(points, points1) == 0

    # test overlapping boundaries
    assert utils.boundary_iou(points, points) == 1

    # test invalid boundaries
    assert utils.boundary_iou(points2, points2) == 0
    assert utils.boundary_iou(points3, points3, zero_division=1) == 1
    assert utils.boundary_iou(points2, points3) == 0


def test_points_center():

    # test unsupported type
    with pytest.raises(AssertionError):
        utils.points_center([1])
    with pytest.raises(AssertionError):
        points = np.array([1, 2, 3])
        utils.points_center(points)

    points = np.array([1, 2, 3, 4])
    assert np.array_equal(utils.points_center(points), np.array([2, 3]))


def test_point_distance():
    # test unsupported type
    with pytest.raises(AssertionError):
        utils.point_distance([1, 2], [1, 2])

    with pytest.raises(AssertionError):
        p = np.array([1, 2, 3])
        utils.point_distance(p, p)

    p = np.array([1, 2])
    assert utils.point_distance(p, p) == 0

    p1 = np.array([2, 2])
    assert utils.point_distance(p, p1) == 1


def test_box_center_distance():
    p1 = np.array([1, 1, 3, 3])
    p2 = np.array([2, 2, 4, 2])

    assert utils.box_center_distance(p1, p2) == 1


def test_box_diag():
    # test unsupported type
    with pytest.raises(AssertionError):
        utils.box_diag([1, 2])
    with pytest.raises(AssertionError):
        utils.box_diag(np.array([1, 2, 3, 4]))

    box = np.array([0, 0, 1, 1, 0, 10, -10, 0])

    assert utils.box_diag(box) == 10


def test_one2one_match_ic13():
    gt_id = 0
    det_id = 0
    recall_mat = np.array([[1, 0], [0, 0]])
    precision_mat = np.array([[1, 0], [0, 0]])
    recall_thr = 0.5
    precision_thr = 0.5
    # test invalid arguments.
    with pytest.raises(AssertionError):
        utils.one2one_match_ic13(0.0, det_id, recall_mat, precision_mat,
                                 recall_thr, precision_thr)
    with pytest.raises(AssertionError):
        utils.one2one_match_ic13(gt_id, 0.0, recall_mat, precision_mat,
                                 recall_thr, precision_thr)
    with pytest.raises(AssertionError):
        utils.one2one_match_ic13(gt_id, det_id, [0, 0], precision_mat,
                                 recall_thr, precision_thr)
    with pytest.raises(AssertionError):
        utils.one2one_match_ic13(gt_id, det_id, recall_mat, [0, 0], recall_thr,
                                 precision_thr)
    with pytest.raises(AssertionError):
        utils.one2one_match_ic13(gt_id, det_id, recall_mat, precision_mat, 1.1,
                                 precision_thr)
    with pytest.raises(AssertionError):
        utils.one2one_match_ic13(gt_id, det_id, recall_mat, precision_mat,
                                 recall_thr, 1.1)

    assert utils.one2one_match_ic13(gt_id, det_id, recall_mat, precision_mat,
                                    recall_thr, precision_thr)
    recall_mat = np.array([[1, 0], [0.6, 0]])
    precision_mat = np.array([[1, 0], [0.6, 0]])
    assert not utils.one2one_match_ic13(
        gt_id, det_id, recall_mat, precision_mat, recall_thr, precision_thr)
    recall_mat = np.array([[1, 0.6], [0, 0]])
    precision_mat = np.array([[1, 0.6], [0, 0]])
    assert not utils.one2one_match_ic13(
        gt_id, det_id, recall_mat, precision_mat, recall_thr, precision_thr)


def test_one2many_match_ic13():
    gt_id = 0
    recall_mat = np.array([[1, 0], [0, 0]])
    precision_mat = np.array([[1, 0], [0, 0]])
    recall_thr = 0.5
    precision_thr = 0.5
    gt_match_flag = [0, 0]
    det_match_flag = [0, 0]
    det_dont_care_index = []
    # test invalid arguments.
    with pytest.raises(AssertionError):
        gt_id_tmp = 0.0
        utils.one2many_match_ic13(gt_id_tmp, recall_mat, precision_mat,
                                  recall_thr, precision_thr, gt_match_flag,
                                  det_match_flag, det_dont_care_index)
    with pytest.raises(AssertionError):
        recall_mat_tmp = [1, 0]
        utils.one2many_match_ic13(gt_id, recall_mat_tmp, precision_mat,
                                  recall_thr, precision_thr, gt_match_flag,
                                  det_match_flag, det_dont_care_index)
    with pytest.raises(AssertionError):
        precision_mat_tmp = [1, 0]
        utils.one2many_match_ic13(gt_id, recall_mat, precision_mat_tmp,
                                  recall_thr, precision_thr, gt_match_flag,
                                  det_match_flag, det_dont_care_index)
    with pytest.raises(AssertionError):

        utils.one2many_match_ic13(gt_id, recall_mat, precision_mat, 1.1,
                                  precision_thr, gt_match_flag, det_match_flag,
                                  det_dont_care_index)
    with pytest.raises(AssertionError):

        utils.one2many_match_ic13(gt_id, recall_mat, precision_mat, recall_thr,
                                  1.1, gt_match_flag, det_match_flag,
                                  det_dont_care_index)
    with pytest.raises(AssertionError):
        gt_match_flag_tmp = np.array([0, 1])
        utils.one2many_match_ic13(gt_id, recall_mat, precision_mat, recall_thr,
                                  precision_thr, gt_match_flag_tmp,
                                  det_match_flag, det_dont_care_index)
    with pytest.raises(AssertionError):
        det_match_flag_tmp = np.array([0, 1])
        utils.one2many_match_ic13(gt_id, recall_mat, precision_mat, recall_thr,
                                  precision_thr, gt_match_flag,
                                  det_match_flag_tmp, det_dont_care_index)
    with pytest.raises(AssertionError):
        det_dont_care_index_tmp = np.array([0, 1])
        utils.one2many_match_ic13(gt_id, recall_mat, precision_mat, recall_thr,
                                  precision_thr, gt_match_flag, det_match_flag,
                                  det_dont_care_index_tmp)

    # test matched case

    result = utils.one2many_match_ic13(gt_id, recall_mat, precision_mat,
                                       recall_thr, precision_thr,
                                       gt_match_flag, det_match_flag,
                                       det_dont_care_index)
    assert result[0]
    assert result[1] == [0]

    # test unmatched case
    gt_match_flag_tmp = [1, 0]
    result = utils.one2many_match_ic13(gt_id, recall_mat, precision_mat,
                                       recall_thr, precision_thr,
                                       gt_match_flag_tmp, det_match_flag,
                                       det_dont_care_index)
    assert not result[0]
    assert result[1] == []


def test_many2one_match_ic13():
    det_id = 0
    recall_mat = np.array([[1, 0], [0, 0]])
    precision_mat = np.array([[1, 0], [0, 0]])
    recall_thr = 0.5
    precision_thr = 0.5
    gt_match_flag = [0, 0]
    det_match_flag = [0, 0]
    gt_dont_care_index = []
    # test invalid arguments.
    with pytest.raises(AssertionError):
        det_id_tmp = 1.0
        utils.many2one_match_ic13(det_id_tmp, recall_mat, precision_mat,
                                  recall_thr, precision_thr, gt_match_flag,
                                  det_match_flag, gt_dont_care_index)
    with pytest.raises(AssertionError):
        recall_mat_tmp = [[1, 0], [0, 0]]
        utils.many2one_match_ic13(det_id, recall_mat_tmp, precision_mat,
                                  recall_thr, precision_thr, gt_match_flag,
                                  det_match_flag, gt_dont_care_index)
    with pytest.raises(AssertionError):
        precision_mat_tmp = [[1, 0], [0, 0]]
        utils.many2one_match_ic13(det_id, recall_mat, precision_mat_tmp,
                                  recall_thr, precision_thr, gt_match_flag,
                                  det_match_flag, gt_dont_care_index)
    with pytest.raises(AssertionError):
        recall_thr_tmp = 1.1
        utils.many2one_match_ic13(det_id, recall_mat, precision_mat,
                                  recall_thr_tmp, precision_thr, gt_match_flag,
                                  det_match_flag, gt_dont_care_index)
    with pytest.raises(AssertionError):
        precision_thr_tmp = 1.1
        utils.many2one_match_ic13(det_id, recall_mat, precision_mat,
                                  recall_thr, precision_thr_tmp, gt_match_flag,
                                  det_match_flag, gt_dont_care_index)
    with pytest.raises(AssertionError):
        gt_match_flag_tmp = np.array([0, 1])
        utils.many2one_match_ic13(det_id, recall_mat, precision_mat,
                                  recall_thr, precision_thr, gt_match_flag_tmp,
                                  det_match_flag, gt_dont_care_index)
    with pytest.raises(AssertionError):
        det_match_flag_tmp = np.array([0, 1])
        utils.many2one_match_ic13(det_id, recall_mat, precision_mat,
                                  recall_thr, precision_thr, gt_match_flag,
                                  det_match_flag_tmp, gt_dont_care_index)
    with pytest.raises(AssertionError):
        gt_dont_care_index_tmp = np.array([0, 1])
        utils.many2one_match_ic13(det_id, recall_mat, precision_mat,
                                  recall_thr, precision_thr, gt_match_flag,
                                  det_match_flag, gt_dont_care_index_tmp)

    # test matched cases

    result = utils.many2one_match_ic13(det_id, recall_mat, precision_mat,
                                       recall_thr, precision_thr,
                                       gt_match_flag, det_match_flag,
                                       gt_dont_care_index)
    assert result[0]
    assert result[1] == [0]

    # test unmatched cases

    gt_dont_care_index = [0]

    result = utils.many2one_match_ic13(det_id, recall_mat, precision_mat,
                                       recall_thr, precision_thr,
                                       gt_match_flag, det_match_flag,
                                       gt_dont_care_index)
    assert not result[0]
    assert result[1] == []
