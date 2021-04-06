import numpy as np

import mmocr.utils as utils
from . import utils as eval_utils


def eval_hmean_iou(pred_boxes,
                   gt_boxes,
                   gt_ignored_boxes,
                   iou_thr=0.5,
                   precision_thr=0.5):
    """Evalute hmean of text detection using IOU standard.

    Args:
        pred_boxes (list[list[list[float]]]): Text boxes for an img list. Each
            box has 2k (>=8) values.
        gt_boxes (list[list[list[float]]]): Ground truth text boxes for an img
            list. Each box has 2k (>=8) values.
        gt_ignored_boxes (list[list[list[float]]]): Ignored ground truth text
            boxes for an img list. Each box has 2k (>=8) values.
        iou_thr (float): Iou threshold when one (gt_box, det_box) pair is
            matched.
        precision_thr (float): Precision threshold when one (gt_box, det_box)
            pair is matched.

    Returns:
        hmean (tuple[dict]): Tuple of dicts indicates the hmean for the dataset
            and all images.
    """
    assert utils.is_3dlist(pred_boxes)
    assert utils.is_3dlist(gt_boxes)
    assert utils.is_3dlist(gt_ignored_boxes)
    assert 0 <= iou_thr <= 1
    assert 0 <= precision_thr <= 1

    img_num = len(pred_boxes)
    assert img_num == len(gt_boxes)
    assert img_num == len(gt_ignored_boxes)

    dataset_gt_num = 0
    dataset_pred_num = 0
    dataset_hit_num = 0

    img_results = []

    for i in range(img_num):
        gt = gt_boxes[i]
        gt_ignored = gt_ignored_boxes[i]
        pred = pred_boxes[i]

        gt_num = len(gt)
        gt_ignored_num = len(gt_ignored)
        pred_num = len(pred)

        hit_num = 0

        # get gt polygons.
        gt_all = gt + gt_ignored
        gt_polys = [eval_utils.points2polygon(p) for p in gt_all]
        gt_ignored_index = [gt_num + i for i in range(len(gt_ignored))]
        gt_num = len(gt_polys)
        pred_polys, _, pred_ignored_index = eval_utils.ignore_pred(
            pred, gt_ignored_index, gt_polys, precision_thr)

        # match.
        if gt_num > 0 and pred_num > 0:
            sz = [gt_num, pred_num]
            iou_mat = np.zeros(sz)

            gt_hit = np.zeros(gt_num, np.int8)
            pred_hit = np.zeros(pred_num, np.int8)

            for gt_id in range(gt_num):
                for pred_id in range(pred_num):
                    gt_pol = gt_polys[gt_id]
                    det_pol = pred_polys[pred_id]

                    iou_mat[gt_id,
                            pred_id] = eval_utils.poly_iou(det_pol, gt_pol)

            for gt_id in range(gt_num):
                for pred_id in range(pred_num):
                    if (gt_hit[gt_id] != 0 or pred_hit[pred_id] != 0
                            or gt_id in gt_ignored_index
                            or pred_id in pred_ignored_index):
                        continue
                    if iou_mat[gt_id, pred_id] > iou_thr:
                        gt_hit[gt_id] = 1
                        pred_hit[pred_id] = 1
                        hit_num += 1

        gt_care_number = gt_num - gt_ignored_num
        pred_care_number = pred_num - len(pred_ignored_index)

        r, p, h = eval_utils.compute_hmean(hit_num, hit_num, gt_care_number,
                                           pred_care_number)

        img_results.append({'recall': r, 'precision': p, 'hmean': h})

        dataset_hit_num += hit_num
        dataset_gt_num += gt_care_number
        dataset_pred_num += pred_care_number

    dataset_r, dataset_p, dataset_h = eval_utils.compute_hmean(
        dataset_hit_num, dataset_hit_num, dataset_gt_num, dataset_pred_num)

    dataset_results = {
        'num_gts': dataset_gt_num,
        'num_dets': dataset_pred_num,
        'num_match': dataset_hit_num,
        'recall': dataset_r,
        'precision': dataset_p,
        'hmean': dataset_h
    }

    return dataset_results, img_results
