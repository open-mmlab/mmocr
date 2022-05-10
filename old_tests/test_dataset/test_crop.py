# Copyright (c) OpenMMLab. All rights reserved.
import math
from itertools import chain, permutations

import numpy as np
import pytest

from mmocr.datasets.pipelines.box_utils import sort_vertex, sort_vertex8
from mmocr.datasets.pipelines.crop import box_jitter, crop_img, warp_img


def test_order_vertex():
    dummy_points_x = [20, 20, 120, 120]
    dummy_points_y = [20, 40, 40, 20]

    expect_points_x = [20, 120, 120, 20]
    expect_points_y = [20, 20, 40, 40]

    with pytest.raises(AssertionError):
        sort_vertex([], dummy_points_y)
    with pytest.raises(AssertionError):
        sort_vertex(dummy_points_x, [])

    for perm in set(permutations([0, 1, 2, 3])):
        points_x = [dummy_points_x[i] for i in perm]
        points_y = [dummy_points_y[i] for i in perm]
        ordered_points_x, ordered_points_y = sort_vertex(points_x, points_y)

        assert np.allclose(ordered_points_x, expect_points_x)
        assert np.allclose(ordered_points_y, expect_points_y)


def test_sort_vertex8():
    dummy_points_x = [21, 21, 122, 122]
    dummy_points_y = [21, 39, 39, 21]

    expect_points = [21, 21, 122, 21, 122, 39, 21, 39]

    for perm in set(permutations([0, 1, 2, 3])):
        points_x = [dummy_points_x[i] for i in perm]
        points_y = [dummy_points_y[i] for i in perm]
        points = list(chain.from_iterable(zip(points_x, points_y)))
        ordered_points = sort_vertex8(points)

        assert np.allclose(ordered_points, expect_points)


def test_box_jitter():
    dummy_points_x = [20, 120, 120, 20]
    dummy_points_y = [20, 20, 40, 40]

    kwargs = dict(jitter_ratio_x=0.0, jitter_ratio_y=0.0)

    with pytest.raises(AssertionError):
        box_jitter([], dummy_points_y)
    with pytest.raises(AssertionError):
        box_jitter(dummy_points_x, [])
    with pytest.raises(AssertionError):
        box_jitter(dummy_points_x, dummy_points_y, jitter_ratio_x=1.)
    with pytest.raises(AssertionError):
        box_jitter(dummy_points_x, dummy_points_y, jitter_ratio_y=1.)

    box_jitter(dummy_points_x, dummy_points_y, **kwargs)

    assert np.allclose(dummy_points_x, [20, 120, 120, 20])
    assert np.allclose(dummy_points_y, [20, 20, 40, 40])


def test_opencv_crop():
    dummy_img = np.ones((600, 600, 3), dtype=np.uint8)
    dummy_box = [20, 20, 120, 20, 120, 40, 20, 40]

    cropped_img = warp_img(dummy_img, dummy_box)

    with pytest.raises(AssertionError):
        warp_img(dummy_img, [])
    with pytest.raises(AssertionError):
        warp_img(dummy_img, [20, 40, 40, 20])

    assert math.isclose(cropped_img.shape[0], 20)
    assert math.isclose(cropped_img.shape[1], 100)


def test_min_rect_crop():
    dummy_img = np.ones((600, 600, 3), dtype=np.uint8)
    dummy_box = [20, 20, 120, 20, 120, 40, 20, 40]

    cropped_img = crop_img(
        dummy_img,
        dummy_box,
        0.,
        0.,
    )

    with pytest.raises(AssertionError):
        crop_img(dummy_img, [])
    with pytest.raises(AssertionError):
        crop_img(dummy_img, [20, 40, 40, 20])
    with pytest.raises(AssertionError):
        crop_img(dummy_img, dummy_box, 4, 0.2)
    with pytest.raises(AssertionError):
        crop_img(dummy_img, dummy_box, 0.4, 1.2)

    assert math.isclose(cropped_img.shape[0], 20)
    assert math.isclose(cropped_img.shape[1], 100)
