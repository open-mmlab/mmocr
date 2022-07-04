# Copyright (c) OpenMMLab. All rights reserved.
"""Test hmean_ic13."""
import math

import pytest

import mmocr.core.evaluation.hmean_ic13 as hmean_ic13
import mmocr.utils as utils


def test_compute_recall_precision():

    gt_polys = []
    det_polys = []

    # test invalid arguments.
    with pytest.raises(AssertionError):
        hmean_ic13.compute_recall_precision(1, 1)

    box1 = [0, 0, 1, 0, 1, 1, 0, 1]

    box2 = [0, 0, 10, 0, 10, 1, 0, 1]

    gt_polys = [utils.poly2shapely(box1)]
    det_polys = [utils.poly2shapely(box2)]
    recall, precision = hmean_ic13.compute_recall_precision(
        gt_polys, det_polys)
    assert recall == 1
    assert precision == 0.1


def test_eval_hmean_ic13():
    det_boxes = []
    gt_boxes = []
    gt_ignored_boxes = []
    precision_thr = 0.4
    recall_thr = 0.8
    center_dist_thr = 1.0
    one2one_score = 1.
    one2many_score = 0.8
    many2one_score = 1
    # test invalid arguments.

    with pytest.raises(AssertionError):
        hmean_ic13.eval_hmean_ic13([1], gt_boxes, gt_ignored_boxes,
                                   precision_thr, recall_thr, center_dist_thr,
                                   one2one_score, one2many_score,
                                   many2one_score)

    with pytest.raises(AssertionError):
        hmean_ic13.eval_hmean_ic13(det_boxes, 1, gt_ignored_boxes,
                                   precision_thr, recall_thr, center_dist_thr,
                                   one2one_score, one2many_score,
                                   many2one_score)
    with pytest.raises(AssertionError):
        hmean_ic13.eval_hmean_ic13(det_boxes, gt_boxes, 1, precision_thr,
                                   recall_thr, center_dist_thr, one2one_score,
                                   one2many_score, many2one_score)
    with pytest.raises(AssertionError):
        hmean_ic13.eval_hmean_ic13(det_boxes, gt_boxes, gt_ignored_boxes, 1.1,
                                   recall_thr, center_dist_thr, one2one_score,
                                   one2many_score, many2one_score)
    with pytest.raises(AssertionError):
        hmean_ic13.eval_hmean_ic13(det_boxes, gt_boxes, gt_ignored_boxes,
                                   precision_thr, 1.1, center_dist_thr,
                                   one2one_score, one2many_score,
                                   many2one_score)
    with pytest.raises(AssertionError):
        hmean_ic13.eval_hmean_ic13(det_boxes, gt_boxes, gt_ignored_boxes,
                                   precision_thr, recall_thr, -1,
                                   one2one_score, one2many_score,
                                   many2one_score)
    with pytest.raises(AssertionError):
        hmean_ic13.eval_hmean_ic13(det_boxes, gt_boxes, gt_ignored_boxes,
                                   precision_thr, recall_thr, center_dist_thr,
                                   -1, one2many_score, many2one_score)
    with pytest.raises(AssertionError):
        hmean_ic13.eval_hmean_ic13(det_boxes, gt_boxes, gt_ignored_boxes,
                                   precision_thr, recall_thr, center_dist_thr,
                                   one2one_score, -1, many2one_score)
    with pytest.raises(AssertionError):
        hmean_ic13.eval_hmean_ic13(det_boxes, gt_boxes, gt_ignored_boxes,
                                   precision_thr, recall_thr, center_dist_thr,
                                   one2one_score, one2many_score, -1)

    # test one2one match
    det_boxes = [[[0, 0, 1, 0, 1, 1, 0, 1], [10, 0, 11, 0, 11, 1, 10, 1]]]
    gt_boxes = [[[0, 0, 1, 0, 1, 1, 0, 1]]]
    gt_ignored_boxes = [[]]
    dataset_result, img_result = hmean_ic13.eval_hmean_ic13(
        det_boxes, gt_boxes, gt_ignored_boxes, precision_thr, recall_thr,
        center_dist_thr, one2one_score, one2many_score, many2one_score)
    assert img_result[0]['recall'] == 1
    assert img_result[0]['precision'] == 0.5
    assert math.isclose(img_result[0]['hmean'], 2 * (0.5) / 1.5)

    # test one2many match
    gt_boxes = [[[0, 0, 2, 0, 2, 1, 0, 1]]]
    det_boxes = [[[0, 0, 1, 0, 1, 1, 0, 1], [1, 0, 2, 0, 2, 1, 1, 1]]]
    dataset_result, img_result = hmean_ic13.eval_hmean_ic13(
        det_boxes, gt_boxes, gt_ignored_boxes, precision_thr, recall_thr,
        center_dist_thr, one2one_score, one2many_score, many2one_score)
    assert img_result[0]['recall'] == 0.8
    assert img_result[0]['precision'] == 1.6 / 2
    assert math.isclose(img_result[0]['hmean'], 2 * (0.64) / 1.6)

    # test many2one match
    precision_thr = 0.6
    recall_thr = 0.8
    det_boxes = [[[0, 0, 2, 0, 2, 1, 0, 1]]]
    gt_boxes = [[[0, 0, 1, 0, 1, 1, 0, 1], [1, 0, 2, 0, 2, 1, 1, 1]]]
    dataset_result, img_result = hmean_ic13.eval_hmean_ic13(
        det_boxes, gt_boxes, gt_ignored_boxes, precision_thr, recall_thr,
        center_dist_thr, one2one_score, one2many_score, many2one_score)
    assert img_result[0]['recall'] == 1
    assert img_result[0]['precision'] == 1
    assert math.isclose(img_result[0]['hmean'], 1)
