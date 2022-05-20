# Copyright (c) OpenMMLab. All rights reserved.
import unittest.mock as mock

import numpy as np
from mmdet.core import PolygonMasks

import mmocr.datasets.pipelines.transforms as transforms


@mock.patch('%s.transforms.np.random.random_sample' % __name__)
def test_scale_aspect_jitter(mock_random):
    img_scale = [(3000, 1000)]  # unused
    ratio_range = (0.5, 1.5)
    aspect_ratio_range = (1, 1)
    multiscale_mode = 'value'
    long_size_bound = 2000
    short_size_bound = 640
    resize_type = 'long_short_bound'
    keep_ratio = False
    jitter = transforms.ScaleAspectJitter(
        img_scale=img_scale,
        ratio_range=ratio_range,
        aspect_ratio_range=aspect_ratio_range,
        multiscale_mode=multiscale_mode,
        long_size_bound=long_size_bound,
        short_size_bound=short_size_bound,
        resize_type=resize_type,
        keep_ratio=keep_ratio)
    mock_random.side_effect = [0.5]

    # test sample_from_range

    result = jitter.sample_from_range([100, 200])
    assert result == 150

    # test _random_scale
    results = {}
    results['img'] = np.zeros((4000, 1000))
    mock_random.side_effect = [0.5, 1]
    jitter._random_scale(results)
    # scale1 0.5， scale2=1 scale =0.5  650/1000, w, h
    # print(results['scale'])
    assert results['scale'] == (650, 2600)


@mock.patch('%s.transforms.np.random.random_sample' % __name__)
def test_square_resize_pad(mock_sample):
    results = {}
    img = np.zeros((15, 30, 3))
    polygon = np.array([10., 5., 20., 5., 20., 10., 10., 10.])
    poly_masks = PolygonMasks([[polygon]], 15, 30)
    results['img'] = img
    results['gt_masks'] = poly_masks
    results['mask_fields'] = ['gt_masks']
    srp = transforms.SquareResizePad(target_size=40, pad_ratio=0.5)

    # test resize with padding
    mock_sample.side_effect = [0.]
    output = srp(results)
    target = 4. / 3 * polygon
    target[1::2] += 10.
    assert np.allclose(output['gt_masks'].masks[0][0], target)
    assert output['img'].shape == (40, 40, 3)

    # test resize to square without padding
    results['img'] = img
    results['gt_masks'] = poly_masks
    mock_sample.side_effect = [1.]
    output = srp(results)
    target = polygon.copy()
    target[::2] *= 4. / 3
    target[1::2] *= 8. / 3
    assert np.allclose(output['gt_masks'].masks[0][0], target)
    assert output['img'].shape == (40, 40, 3)
