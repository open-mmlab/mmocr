# Copyright (c) OpenMMLab. All rights reserved.
# TODO check whether to keep these utils after refactoring ic13 metrics


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
