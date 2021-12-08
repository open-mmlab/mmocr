# Copyright (c) OpenMMLab. All rights reserved.
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

    loader = LoadTextAnnotations(with_bbox, with_label, with_mask, with_seg,
                                 poly2mask)
    results['ori_shape'] = (640, 640, 3)
    output = loader._load_masks(results)
    assert len(output['gt_masks_ignore']) == 4
    assert np.allclose(output['gt_masks_ignore'].masks[0],
                       [[499, 94, 531, 94, 531, 124, 499, 124]])
    loader = LoadTextAnnotations(with_bbox, with_label, with_mask, with_seg,
                                 True)
    output = loader._load_masks(results)


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
