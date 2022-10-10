# Copyright (c) OpenMMLab. All rights reserved.
import math
import unittest.mock as mock

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

import mmocr.datasets.pipelines.ocr_transforms as transforms


def test_resize_ocr():
    input_img = np.ones((64, 256, 3), dtype=np.uint8)

    rci = transforms.ResizeOCR(
        32, min_width=32, max_width=160, keep_aspect_ratio=True)
    results = {'img_shape': input_img.shape, 'img': input_img}

    # test call
    results = rci(results)
    assert np.allclose([32, 160, 3], results['pad_shape'])
    assert np.allclose([32, 160, 3], results['img'].shape)
    assert 'valid_ratio' in results
    assert math.isclose(results['valid_ratio'], 0.8)
    assert math.isclose(np.sum(results['img'][:, 129:, :]), 0)

    rci = transforms.ResizeOCR(
        32, min_width=32, max_width=160, keep_aspect_ratio=False)
    results = {'img_shape': input_img.shape, 'img': input_img}
    results = rci(results)
    assert math.isclose(results['valid_ratio'], 1)

    # test img_pad_value
    rci = transforms.ResizeOCR(
        32, min_width=32, max_width=160, keep_aspect_ratio=True,
        img_pad_value=(127, 127, 127))
    results = {'img_shape': input_img.shape, 'img': input_img}
    results = rci(results)
    assert results['img'].shape == (32, 160, 3)
    assert np.all(results['img'][:32,:128,:] == np.array([1]))
    assert np.all(results['img'][:32,128:,:] == np.array([127]))
    

def test_to_tensor():
    input_img = np.ones((64, 256, 3), dtype=np.uint8)

    expect_output = TF.to_tensor(input_img)
    rci = transforms.ToTensorOCR()

    results = {'img': input_img}
    results = rci(results)

    assert np.allclose(results['img'].numpy(), expect_output.numpy())


def test_normalize():
    inputs = torch.zeros(3, 10, 10)

    expect_output = torch.ones_like(inputs) * (-1)
    rci = transforms.NormalizeOCR(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    results = {'img': inputs}
    results = rci(results)

    assert np.allclose(results['img'].numpy(), expect_output.numpy())


@mock.patch('%s.transforms.np.random.random' % __name__)
def test_online_crop(mock_random):
    kwargs = dict(
        box_keys=['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'],
        jitter_prob=0.5,
        max_jitter_ratio_x=0.05,
        max_jitter_ratio_y=0.02)

    mock_random.side_effect = [0.1, 1, 1, 1]

    src_img = np.ones((100, 100, 3), dtype=np.uint8)
    results = {
        'img': src_img,
        'img_info': {
            'x1': '20',
            'y1': '20',
            'x2': '40',
            'y2': '20',
            'x3': '40',
            'y3': '40',
            'x4': '20',
            'y4': '40'
        }
    }

    rci = transforms.OnlineCropOCR(**kwargs)

    results = rci(results)

    assert np.allclose(results['img_shape'], [20, 20, 3])

    # test not crop
    mock_random.side_effect = [0.1, 1, 1, 1]
    results['img_info'] = {}
    results['img'] = src_img

    results = rci(results)
    assert np.allclose(results['img'].shape, [100, 100, 3])


def test_fancy_pca():
    input_tensor = torch.rand(3, 32, 100)

    rci = transforms.FancyPCA()

    results = {'img': input_tensor}
    results = rci(results)

    assert results['img'].shape == torch.Size([3, 32, 100])


@mock.patch('%s.transforms.np.random.uniform' % __name__)
def test_random_padding(mock_random):
    kwargs = dict(max_ratio=[0.0, 0.0, 0.0, 0.0], box_type=None)

    mock_random.side_effect = [1, 1, 1, 1]

    src_img = np.ones((32, 100, 3), dtype=np.uint8)
    results = {'img': src_img, 'img_shape': (32, 100, 3)}

    rci = transforms.RandomPaddingOCR(**kwargs)

    results = rci(results)
    print(results['img'].shape)
    assert np.allclose(results['img_shape'], [96, 300, 3])


def test_opencv2pil():
    src_img = np.ones((32, 100, 3), dtype=np.uint8)
    results = {'img': src_img}
    rci = transforms.OpencvToPil()

    results = rci(results)
    assert np.allclose(results['img'].size, (100, 32))


def test_pil2opencv():
    src_img = Image.new('RGB', (100, 32), color=(255, 255, 255))
    results = {'img': src_img}
    rci = transforms.PilToOpencv()

    results = rci(results)
    assert np.allclose(results['img'].shape, (32, 100, 3))
