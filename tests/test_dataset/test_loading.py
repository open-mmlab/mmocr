# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np

from mmocr.datasets.pipelines import LoadImageFromNdarray, LoadTextAnnotations


def _create_dummy_ann():
    results = {}
    results['img_info'] = {}
    results['img_info']['height'] = 1000
    results['img_info']['width'] = 1000
    results['ann_info'] = {}
    results['ann_info']['masks'] = []
    results['mask_fields'] = []
    results['ann_info']['masks_ignore'] = [
        [[499, 94, 531, 94, 531, 124, 499, 124]],
        [[3, 156, 81, 155, 78, 181, 0, 182]],
        [[11, 223, 59, 221, 59, 234, 11, 236]],
        [[500, 156, 551, 156, 550, 165, 499, 165]]
    ]

    return results


def test_loadtextannotation():

    results = _create_dummy_ann()
    with_bbox = True
    with_label = True
    with_mask = True
    with_seg = False
    poly2mask = False

    # If no 'ori_shape' in result but use_img_shape=True,
    # result['img_info']['height'] and result['img_info']['width']
    # will be used to generate mask.
    loader = LoadTextAnnotations(
        with_bbox,
        with_label,
        with_mask,
        with_seg,
        poly2mask,
        use_img_shape=True)
    tmp_results = copy.deepcopy(results)
    output = loader._load_masks(tmp_results)
    assert len(output['gt_masks_ignore']) == 4
    assert np.allclose(output['gt_masks_ignore'].masks[0],
                       [[499, 94, 531, 94, 531, 124, 499, 124]])
    assert output['gt_masks_ignore'].height == results['img_info']['height']

    # If 'ori_shape' in result and use_img_shape=True,
    # result['ori_shape'] will be used to generate mask.
    loader = LoadTextAnnotations(
        with_bbox,
        with_label,
        with_mask,
        with_seg,
        poly2mask=True,
        use_img_shape=True)
    tmp_results = copy.deepcopy(results)
    tmp_results['ori_shape'] = (640, 640, 3)
    output = loader._load_masks(tmp_results)
    assert output['img_info']['height'] == 640
    assert output['gt_masks_ignore'].height == 640


def test_load_img_from_numpy():
    result = {'img': np.ones((32, 100, 3), dtype=np.uint8)}

    load = LoadImageFromNdarray(color_type='color')
    output = load(result)

    assert output['img'].shape[2] == 3
    assert len(output['img'].shape) == 3

    result = {'img': np.ones((32, 100, 1), dtype=np.uint8)}
    load = LoadImageFromNdarray(color_type='color')
    output = load(result)
    assert output['img'].shape[2] == 3

    result = {'img': np.ones((32, 100, 3), dtype=np.uint8)}
    load = LoadImageFromNdarray(color_type='grayscale', to_float32=True)
    output = load(result)
    assert output['img'].shape[2] == 1
