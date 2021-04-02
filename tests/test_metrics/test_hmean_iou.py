"""Test hmean_iou."""
import pytest

import mmocr.core.evaluation.hmean_iou as hmean_iou


def test_eval_hmean_iou():

    pred_boxes = []
    gt_boxes = []
    gt_ignored_boxes = []
    iou_thr = 0.5
    precision_thr = 0.5

    # test invalid arguments.

    with pytest.raises(AssertionError):
        hmean_iou.eval_hmean_iou([1], gt_boxes, gt_ignored_boxes, iou_thr,
                                 precision_thr)
    with pytest.raises(AssertionError):
        hmean_iou.eval_hmean_iou(pred_boxes, [1], gt_ignored_boxes, iou_thr,
                                 precision_thr)
    with pytest.raises(AssertionError):
        hmean_iou.eval_hmean_iou(pred_boxes, gt_boxes, [1], iou_thr,
                                 precision_thr)
    with pytest.raises(AssertionError):
        hmean_iou.eval_hmean_iou(pred_boxes, gt_boxes, gt_ignored_boxes, 1.1,
                                 precision_thr)
    with pytest.raises(AssertionError):
        hmean_iou.eval_hmean_iou(pred_boxes, gt_boxes, gt_ignored_boxes,
                                 iou_thr, 1.1)

    pred_boxes = [[[0, 0, 1, 0, 1, 1, 0, 1], [2, 0, 3, 0, 3, 1, 2, 1]]]
    gt_boxes = [[[0, 0, 1, 0, 1, 1, 0, 1], [2, 0, 3, 0, 3, 1, 2, 1]]]
    gt_ignored_boxes = [[]]
    results = hmean_iou.eval_hmean_iou(pred_boxes, gt_boxes, gt_ignored_boxes,
                                       iou_thr, precision_thr)
    assert results[1][0]['recall'] == 1
    assert results[1][0]['precision'] == 1
    assert results[1][0]['hmean'] == 1
