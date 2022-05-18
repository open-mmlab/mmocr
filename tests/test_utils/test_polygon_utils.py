# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch

from mmocr.utils import rescale_polygon, rescale_polygons
from mmocr.utils.polygon_utils import crop_polygon


class TestCropPolygon(unittest.TestCase):

    def test_crop_polygon(self):
        # polygon cross box
        polygon = np.array([20., -10., 40., 10., 10., 40., -10., 20.])
        crop_box = np.array([0., 0., 60., 60.])
        target_poly_cropped = np.array([[10., 40., 30., 10., 0., 0., 10.],
                                        [40., 10., 0., 0., 10., 30., 40.]])
        poly_cropped = crop_polygon(polygon, crop_box)
        self.assertTrue(target_poly_cropped.all() == poly_cropped.all())

        # polygon inside box
        polygon = np.array([0., 0., 30., 0., 30., 30., 0., 30.]).reshape(-1, 2)
        crop_box = np.array([0., 0., 60., 60.])
        target_poly_cropped = polygon
        poly_cropped = crop_polygon(polygon, crop_box)
        self.assertTrue(target_poly_cropped.all() == poly_cropped.all())

        # polygon outside box
        polygon = np.array([0., 0., 30., 0., 30., 30., 0., 30.]).reshape(-1, 2)
        crop_box = np.array([80., 80., 90., 90.])
        target_poly_cropped = polygon
        poly_cropped = crop_polygon(polygon, crop_box)
        self.assertEqual(poly_cropped, None)


class TestPolygonUtils(unittest.TestCase):

    def test_rescale_polygon(self):
        scale_factor = (0.3, 0.4)

        with self.assertRaises(AssertionError):
            polygons = [0, 0, 1, 0, 1, 1, 0]
            rescale_polygon(polygons, scale_factor)

        polygons = [0, 0, 1, 0, 1, 1, 0, 1]
        self.assertTrue(
            np.allclose(
                rescale_polygon(polygons, scale_factor, mode='div'),
                np.array([0, 0, 1 / 0.3, 0, 1 / 0.3, 1 / 0.4, 0, 1 / 0.4])))
        self.assertTrue(
            np.allclose(
                rescale_polygon(polygons, scale_factor, mode='mul'),
                np.array([0, 0, 0.3, 0, 0.3, 0.4, 0, 0.4])))

    def test_rescale_polygons(self):
        polygons = [
            np.array([0, 0, 1, 0, 1, 1, 0, 1]),
            np.array([1, 1, 2, 1, 2, 2, 1, 2])
        ]
        scale_factor = (0.5, 0.5)
        self.assertTrue(
            np.allclose(
                rescale_polygons(polygons, scale_factor, mode='div'), [
                    np.array([0, 0, 2, 0, 2, 2, 0, 2]),
                    np.array([2, 2, 4, 2, 4, 4, 2, 4])
                ]))
        self.assertTrue(
            np.allclose(
                rescale_polygons(polygons, scale_factor, mode='mul'), [
                    np.array([0, 0, 0.5, 0, 0.5, 0.5, 0, 0.5]),
                    np.array([0.5, 0.5, 1, 0.5, 1, 1, 0.5, 1])
                ]))

        polygons = [torch.Tensor([0, 0, 1, 0, 1, 1, 0, 1])]
        scale_factor = (0.3, 0.4)
        self.assertTrue(
            np.allclose(
                rescale_polygons(polygons, scale_factor, mode='div'),
                [np.array([0, 0, 1 / 0.3, 0, 1 / 0.3, 1 / 0.4, 0, 1 / 0.4])]))
        self.assertTrue(
            np.allclose(
                rescale_polygons(polygons, scale_factor, mode='mul'),
                [np.array([0, 0, 0.3, 0, 0.3, 0.4, 0, 0.4])]))
