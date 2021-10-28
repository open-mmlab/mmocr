# Copyright (c) OpenMMLab. All rights reserved.
"""Tests the utils of evaluation."""
import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon

import mmocr.core.evaluation.utils as utils


def test_ignore_pred():

    # test invalid arguments
    box = [0, 0, 1, 0, 1, 1, 0, 1]
    det_boxes = [box]
    gt_dont_care_index = [0]
    gt_polys = [utils.points2polygon(box)]
    precision_thr = 0.5

    with pytest.raises(AssertionError):
        det_boxes_tmp = 1
        utils.ignore_pred(det_boxes_tmp, gt_dont_care_index, gt_polys,
                          precision_thr)
    with pytest.raises(AssertionError):
        gt_dont_care_index_tmp = 1
        utils.ignore_pred(det_boxes, gt_dont_care_index_tmp, gt_polys,
                          precision_thr)
    with pytest.raises(AssertionError):
        gt_polys_tmp = 1
        utils.ignore_pred(det_boxes, gt_dont_care_index, gt_polys_tmp,
                          precision_thr)
    with pytest.raises(AssertionError):
        precision_thr_tmp = 1.1
        utils.ignore_pred(det_boxes, gt_dont_care_index, gt_polys,
                          precision_thr_tmp)

    # test ignored cases
    result = utils.ignore_pred(det_boxes, gt_dont_care_index, gt_polys,
                               precision_thr)
    assert result[2] == [0]
    # test unignored cases
    gt_dont_care_index_tmp = []
    result = utils.ignore_pred(det_boxes, gt_dont_care_index_tmp, gt_polys,
                               precision_thr)
    assert result[2] == []

    det_boxes_tmp = [[10, 10, 15, 10, 15, 15, 10, 15]]
    result = utils.ignore_pred(det_boxes_tmp, gt_dont_care_index, gt_polys,
                               precision_thr)
    assert result[2] == []


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


def test_points2polygon():

    # test unsupported type
    with pytest.raises(AssertionError):
        points = 2
        utils.points2polygon(points)

    # test unsupported size
    with pytest.raises(AssertionError):
        points = [1, 2, 3, 4, 5, 6, 7]
        utils.points2polygon(points)
    with pytest.raises(AssertionError):
        points = [1, 2, 3, 4, 5, 6]
        utils.points2polygon(points)

    # test np.array
    points = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    poly = utils.points2polygon(points)
    i = 0
    for coord in poly.exterior.coords[:-1]:
        assert coord[0] == points[i]
        assert coord[1] == points[i + 1]
        i += 2

    points = [1, 2, 3, 4, 5, 6, 7, 8]
    poly = utils.points2polygon(points)
    i = 0
    for coord in poly.exterior.coords[:-1]:
        assert coord[0] == points[i]
        assert coord[1] == points[i + 1]
        i += 2


def test_poly_intersection():

    # test unsupported type
    with pytest.raises(AssertionError):
        utils.poly_intersection(0, 1)

    # test non-overlapping polygons

    points = [0, 0, 0, 1, 1, 1, 1, 0]
    points1 = [10, 20, 30, 40, 50, 60, 70, 80]
    points2 = [0, 0, 0, 0, 0, 0, 0, 0]  # Invalid polygon
    points3 = [0, 0, 0, 1, 1, 0, 1, 1]  # Self-intersected polygon
    points4 = [0.5, 0, 1.5, 0, 1.5, 1, 0.5, 1]
    poly = utils.points2polygon(points)
    poly1 = utils.points2polygon(points1)
    poly2 = utils.points2polygon(points2)
    poly3 = utils.points2polygon(points3)
    poly4 = utils.points2polygon(points4)

    area_inters = utils.poly_intersection(poly, poly1)

    assert area_inters == 0

    # test overlapping polygons
    area_inters = utils.poly_intersection(poly, poly)
    assert area_inters == 1
    area_inters = utils.poly_intersection(poly, poly4)
    assert area_inters == 0.5

    # test invalid polygons
    assert utils.poly_intersection(poly2, poly2) == 0
    assert utils.poly_intersection(poly3, poly3, invalid_ret=1) == 1
    # The return value depends on the implementation of the package
    assert utils.poly_intersection(poly3, poly3, invalid_ret=None) == 0.25

    # test poly return
    _, poly = utils.poly_intersection(poly, poly4, return_poly=True)
    assert isinstance(poly, Polygon)
    _, poly = utils.poly_intersection(
        poly3, poly3, invalid_ret=None, return_poly=True)
    assert isinstance(poly, Polygon)
    _, poly = utils.poly_intersection(
        poly2, poly3, invalid_ret=1, return_poly=True)
    assert poly is None


def test_poly_union():

    # test unsupported type
    with pytest.raises(AssertionError):
        utils.poly_union(0, 1)

    # test non-overlapping polygons

    points = [0, 0, 0, 1, 1, 1, 1, 0]
    points1 = [2, 2, 2, 3, 3, 3, 3, 2]
    points2 = [0, 0, 0, 0, 0, 0, 0, 0]  # Invalid polygon
    points3 = [0, 0, 0, 1, 1, 0, 1, 1]  # Self-intersected polygon
    points4 = [0.5, 0.5, 1, 0, 1, 1, 0.5, 0.5]
    poly = utils.points2polygon(points)
    poly1 = utils.points2polygon(points1)
    poly2 = utils.points2polygon(points2)
    poly3 = utils.points2polygon(points3)
    poly4 = utils.points2polygon(points4)

    assert utils.poly_union(poly, poly1) == 2

    # test overlapping polygons
    assert utils.poly_union(poly, poly) == 1

    # test invalid polygons
    assert utils.poly_union(poly2, poly2) == 0
    assert utils.poly_union(poly3, poly3, invalid_ret=1) == 1

    # The return value depends on the implementation of the package
    assert utils.poly_union(poly3, poly3, invalid_ret=None) == 0.25
    assert utils.poly_union(poly2, poly3) == 0.25
    assert utils.poly_union(poly3, poly4) == 0.5

    # test poly return
    _, poly = utils.poly_union(poly, poly1, return_poly=True)
    assert isinstance(poly, MultiPolygon)
    _, poly = utils.poly_union(poly3, poly3, return_poly=True)
    assert isinstance(poly, Polygon)
    _, poly = utils.poly_union(poly2, poly3, invalid_ret=0, return_poly=True)
    assert poly is None


def test_poly_iou():

    # test unsupported type
    with pytest.raises(AssertionError):
        utils.poly_iou([1], [2])

    points = [0, 0, 0, 1, 1, 1, 1, 0]
    points1 = [10, 20, 30, 40, 50, 60, 70, 80]
    points2 = [0, 0, 0, 0, 0, 0, 0, 0]  # Invalid polygon
    points3 = [0, 0, 0, 1, 1, 0, 1, 1]  # Self-intersected polygon

    poly = utils.points2polygon(points)
    poly1 = utils.points2polygon(points1)
    poly2 = utils.points2polygon(points2)
    poly3 = utils.points2polygon(points3)

    assert utils.poly_iou(poly, poly1) == 0

    # test overlapping polygons
    assert utils.poly_iou(poly, poly) == 1

    # test invalid polygons
    assert utils.poly_iou(poly2, poly2) == 0
    assert utils.poly_iou(poly3, poly3, zero_division=1) == 1
    assert utils.poly_iou(poly2, poly3) == 0


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
