import tempfile

import numpy as np
import pytest

from mmocr.core.evaluation.hmean import (eval_hmean, get_gt_masks,
                                         output_ranklist)


def _create_dummy_ann_infos():
    ann_infos = {
        'bboxes': np.array([[50., 70., 80., 100.]], dtype=np.float32),
        'labels': np.array([1], dtype=np.int64),
        'bboxes_ignore': np.array([[120, 140, 200, 200]], dtype=np.float32),
        'masks': [[[50, 70, 80, 70, 80, 100, 50, 100]]],
        'masks_ignore': [[[120, 140, 200, 140, 200, 200, 120, 200]]]
    }
    return [ann_infos]


def test_output_ranklist():
    result = [{'hmean': 1}, {'hmean': 0.5}]
    file_name = tempfile.NamedTemporaryFile().name
    img_infos = [{'file_name': 'sample1.jpg'}, {'file_name': 'sample2.jpg'}]

    json_file = file_name + '.json'
    with pytest.raises(AssertionError):
        output_ranklist([[]], img_infos, json_file)
    with pytest.raises(AssertionError):
        output_ranklist(result, [[]], json_file)
    with pytest.raises(AssertionError):
        output_ranklist(result, img_infos, file_name)

    sorted_outputs = output_ranklist(result, img_infos, json_file)

    assert sorted_outputs[0]['hmean'] == 0.5


def test_get_gt_mask():
    ann_infos = _create_dummy_ann_infos()
    gt_masks, gt_masks_ignore = get_gt_masks(ann_infos)

    assert np.allclose(gt_masks[0], [[50, 70, 80, 70, 80, 100, 50, 100]])
    assert np.allclose(gt_masks_ignore[0],
                       [[120, 140, 200, 140, 200, 200, 120, 200]])


def test_eval_hmean():
    metrics = set(['hmean-iou', 'hmean-ic13'])
    results = [{
        'boundary_result': [[50, 70, 80, 70, 80, 100, 50, 100, 1],
                            [120, 140, 200, 140, 200, 200, 120, 200, 1]]
    }]

    img_infos = [{'file_name': 'sample1.jpg'}]
    ann_infos = _create_dummy_ann_infos()

    # test invalid arguments
    with pytest.raises(AssertionError):
        eval_hmean(results, [[]], ann_infos, metrics=metrics)
    with pytest.raises(AssertionError):
        eval_hmean(results, img_infos, [[]], metrics=metrics)
    with pytest.raises(AssertionError):
        eval_hmean([[]], img_infos, ann_infos, metrics=metrics)
    with pytest.raises(AssertionError):
        eval_hmean(results, img_infos, ann_infos, metrics='hmean-iou')

    eval_results = eval_hmean(results, img_infos, ann_infos, metrics=metrics)

    assert eval_results['hmean-iou:hmean'] == 1
    assert eval_results['hmean-ic13:hmean'] == 1
