# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest.mock as mock

import numpy as np
import pytest

from mmocr.datasets.pipelines import (OneOfWrapper, RandomWrapper,
                                      TorchVisionWrapper)
from mmocr.datasets.pipelines.transforms import ColorJitter


def test_torchvision_wrapper():
    x = {'img': np.ones((128, 100, 3), dtype=np.uint8)}
    # object not found error
    with pytest.raises(Exception):
        TorchVisionWrapper(op='NonExist')
    with pytest.raises(TypeError):
        TorchVisionWrapper()
    f = TorchVisionWrapper('Grayscale')
    with pytest.raises(AssertionError):
        f({})
    results = f(x)
    assert results['img'].shape == (128, 100)
    assert results['img_shape'] == (128, 100)


@mock.patch('random.choice')
def test_oneof(rand_choice):
    color_jitter = dict(type='TorchVisionWrapper', op='ColorJitter')
    gray_scale = dict(type='TorchVisionWrapper', op='Grayscale')
    x = {'img': np.random.randint(0, 256, size=(128, 100, 3), dtype=np.uint8)}
    f = OneOfWrapper([color_jitter, gray_scale])
    # Use color_jitter at the first call
    rand_choice.side_effect = lambda x: x[0]
    results = f(x)
    assert results['img'].shape == (128, 100, 3)
    # Use gray_scale at the second call
    rand_choice.side_effect = lambda x: x[1]
    results = f(x)
    assert results['img'].shape == (128, 100)

    # Passing object
    f = OneOfWrapper([ColorJitter(), gray_scale])
    # Use color_jitter at the first call
    results = f(x)
    assert results['img'].shape == (128, 100)

    # Test invalid inputs
    with pytest.raises(AssertionError):
        f = OneOfWrapper(None)
    with pytest.raises(AssertionError):
        f = OneOfWrapper([])
    with pytest.raises(AssertionError):
        f = OneOfWrapper({})


@mock.patch('numpy.random.uniform')
def test_runwithprob(np_random_uniform):
    np_random_uniform.side_effect = [0.1, 0.9]
    f = RandomWrapper([dict(type='TorchVisionWrapper', op='Grayscale')], 0.5)
    img = np.random.randint(0, 256, size=(128, 100, 3), dtype=np.uint8)
    results = f({'img': copy.deepcopy(img)})
    assert results['img'].shape == (128, 100)
    results = f({'img': copy.deepcopy(img)})
    assert results['img'].shape == (128, 100, 3)
