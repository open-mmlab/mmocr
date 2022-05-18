# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch

from mmocr.utils import rescale_polygon, rescale_polygons


class TestPolygonUtils(unittest.TestCase):

    def test_rescale_polygon(self):
        scale_factor = (0.3, 0.4)

        with self.assertRaises(AssertionError):
            polygons = [0, 0, 1, 0, 1, 1, 0]
            rescale_polygon(polygons, scale_factor)

        polygons = [0, 0, 1, 0, 1, 1, 0, 1]
        self.assertTrue(
            np.allclose(
                rescale_polygon(polygons, scale_factor),
                np.array([0, 0, 1 / 0.3, 0, 1 / 0.3, 1 / 0.4, 0, 1 / 0.4])))

    def test_rescale_polygons(self):
        polygons = [
            np.array([0, 0, 1, 0, 1, 1, 0, 1]),
            np.array([1, 1, 2, 1, 2, 2, 1, 2])
        ]
        scale_factor = (0.5, 0.5)
        self.assertTrue(
            np.allclose(
                rescale_polygons(polygons, scale_factor), [
                    np.array([0, 0, 2, 0, 2, 2, 0, 2]),
                    np.array([2, 2, 4, 2, 4, 4, 2, 4])
                ]))

        polygons = [torch.Tensor([0, 0, 1, 0, 1, 1, 0, 1])]
        scale_factor = (0.3, 0.4)
        self.assertTrue(
            np.allclose(
                rescale_polygons(polygons, scale_factor),
                [np.array([0, 0, 1 / 0.3, 0, 1 / 0.3, 1 / 0.4, 0, 1 / 0.4])]))
