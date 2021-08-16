# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from mmocr.datasets.pipelines.test_time_aug import MultiRotateAugOCR


def test_resize_ocr():
    input_img1 = np.ones((64, 256, 3), dtype=np.uint8)
    input_img2 = np.ones((64, 32, 3), dtype=np.uint8)

    rci = MultiRotateAugOCR(transforms=[], rotate_degrees=[0, 90, 270])

    # test invalid arguments
    with pytest.raises(AssertionError):
        MultiRotateAugOCR(transforms=[], rotate_degrees=[45])
    with pytest.raises(AssertionError):
        MultiRotateAugOCR(transforms=[], rotate_degrees=[20.5])

    # test call with input_img1
    results = {'img_shape': input_img1.shape, 'img': input_img1}
    results = rci(results)
    assert np.allclose([64, 256, 3], results['img_shape'])
    assert len(results['img']) == 1
    assert len(results['img_shape']) == 1
    assert np.allclose([64, 256, 3], results['img_shape'][0])

    # test call with input_img2
    results = {'img_shape': input_img2.shape, 'img': input_img2}
    results = rci(results)
    assert np.allclose([64, 32, 3], results['img_shape'])
    assert len(results['img']) == 3
    assert len(results['img_shape']) == 3
    assert np.allclose([64, 32, 3], results['img_shape'][0])
