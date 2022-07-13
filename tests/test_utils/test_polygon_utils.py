# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch
from shapely.geometry import MultiPolygon, Polygon

from mmocr.utils import (boundary_iou, crop_polygon, offset_polygon, poly2bbox,
                         poly2shapely, poly_intersection, poly_iou,
                         poly_make_valid, poly_union, polys2shapely,
                         rescale_polygon, rescale_polygons, shapely2poly)


class TestCropPolygon(unittest.TestCase):

    def test_crop_polygon(self):
        # polygon cross box
        polygon = np.array([20., -10., 40., 10., 10., 40., -10., 20.])
        crop_box = np.array([0., 0., 60., 60.])
        target_poly_cropped = np.array(
            [10, 40, 0, 30, 0, 10, 10, 0, 30, 0, 40, 10])
        poly_cropped = crop_polygon(polygon, crop_box)
        self.assertTrue(
            poly2shapely(poly_cropped).equals(
                poly2shapely(target_poly_cropped)))

        # polygon inside box
        polygon = np.array([0., 0., 30., 0., 30., 30., 0., 30.])
        crop_box = np.array([0., 0., 60., 60.])
        target_poly_cropped = polygon
        poly_cropped = crop_polygon(polygon, crop_box)
        self.assertTrue(
            poly2shapely(poly_cropped).equals(
                poly2shapely(target_poly_cropped)))

        # polygon outside box
        polygon = np.array([0., 0., 30., 0., 30., 30., 0., 30.])
        crop_box = np.array([80., 80., 90., 90.])
        poly_cropped = crop_polygon(polygon, crop_box)
        self.assertEqual(poly_cropped, None)

        # polygon and box are overlapped at a point
        polygon = np.array([0., 0., 10., 0., 10., 10., 0., 10.])
        crop_box = np.array([10., 10., 20., 20.])
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

    def test_poly2bbox(self):
        # test np.array
        polygon = np.array([0, 0, 1, 0, 1, 1, 0, 1])
        self.assertTrue(np.all(poly2bbox(polygon) == np.array([0, 0, 1, 1])))
        # test list
        polygon = [0, 0, 1, 0, 1, 1, 0, 1]
        self.assertTrue(np.all(poly2bbox(polygon) == np.array([0, 0, 1, 1])))
        # test tensor
        polygon = torch.Tensor([0, 0, 1, 0, 1, 1, 0, 1])
        self.assertTrue(np.all(poly2bbox(polygon) == np.array([0, 0, 1, 1])))

    def test_poly2shapely(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        # test np.array
        poly = np.array([0, 0, 1, 0, 1, 1, 0, 1])
        self.assertEqual(poly2shapely(poly), polygon)
        # test list
        poly = [0, 0, 1, 0, 1, 1, 0, 1]
        self.assertEqual(poly2shapely(poly), polygon)
        # test tensor
        poly = torch.Tensor([0, 0, 1, 0, 1, 1, 0, 1])
        self.assertEqual(poly2shapely(poly), polygon)
        # test invalid
        poly = [0, 0, 1]
        with self.assertRaises(AssertionError):
            poly2shapely(poly)
        poly = [0, 0, 1, 0, 1, 1, 0, 1, 1]
        with self.assertRaises(AssertionError):
            poly2shapely(poly)

    def test_polys2shapely(self):
        polygons = [
            Polygon([[0, 0], [1, 0], [1, 1], [0, 1]]),
            Polygon([[1, 0], [1, 1], [0, 1], [0, 0]])
        ]
        # test np.array
        polys = np.array([[0, 0, 1, 0, 1, 1, 0, 1], [1, 0, 1, 1, 0, 1, 0, 0]])
        self.assertEqual(polys2shapely(polys), polygons)
        # test list
        polys = [[0, 0, 1, 0, 1, 1, 0, 1], [1, 0, 1, 1, 0, 1, 0, 0]]
        self.assertEqual(polys2shapely(polys), polygons)
        # test tensor
        polys = torch.Tensor([[0, 0, 1, 0, 1, 1, 0, 1],
                              [1, 0, 1, 1, 0, 1, 0, 0]])
        self.assertEqual(polys2shapely(polys), polygons)
        # test invalid
        polys = [0, 0, 1]
        with self.assertRaises(AssertionError):
            polys2shapely(polys)
        polys = [0, 0, 1, 0, 1, 1, 0, 1, 1]
        with self.assertRaises(AssertionError):
            polys2shapely(polys)

    def test_shapely2poly(self):
        polygon = Polygon([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        poly = np.array([0., 0., 1., 0., 1., 1., 0., 1., 0., 0.])
        self.assertTrue(poly2shapely(poly).equals(polygon))
        self.assertTrue(isinstance(shapely2poly(polygon), np.ndarray))

    def test_poly_make_valid(self):
        poly = Polygon([[0, 0], [1, 1], [1, 0], [0, 1]])
        self.assertFalse(poly.is_valid)
        poly = poly_make_valid(poly)
        self.assertTrue(poly.is_valid)
        # invalid input
        with self.assertRaises(AssertionError):
            poly_make_valid([0, 0, 1, 1, 1, 0, 0, 1])

    def test_poly_intersection(self):

        # test unsupported type
        with self.assertRaises(AssertionError):
            poly_intersection(0, 1)

        # test non-overlapping polygons
        points = [0, 0, 0, 1, 1, 1, 1, 0]
        points1 = [10, 20, 30, 40, 50, 60, 70, 80]
        points2 = [0, 0, 0, 0, 0, 0, 0, 0]  # Invalid polygon
        points3 = [0, 0, 0, 1, 1, 0, 1, 1]  # Self-intersected polygon
        points4 = [0.5, 0, 1.5, 0, 1.5, 1, 0.5, 1]
        poly = poly2shapely(points)
        poly1 = poly2shapely(points1)
        poly2 = poly2shapely(points2)
        poly3 = poly2shapely(points3)
        poly4 = poly2shapely(points4)

        area_inters = poly_intersection(poly, poly1)
        self.assertEqual(area_inters, 0.)

        # test overlapping polygons
        area_inters = poly_intersection(poly, poly)
        self.assertEqual(area_inters, 1)
        area_inters = poly_intersection(poly, poly4)
        self.assertEqual(area_inters, 0.5)

        # test invalid polygons
        self.assertEqual(poly_intersection(poly2, poly2), 0)
        self.assertEqual(poly_intersection(poly3, poly3, invalid_ret=1), 1)
        self.assertEqual(
            poly_intersection(poly3, poly3, invalid_ret=None), 0.25)

        # test poly return
        _, poly = poly_intersection(poly, poly4, return_poly=True)
        self.assertTrue(isinstance(poly, Polygon))
        _, poly = poly_intersection(
            poly3, poly3, invalid_ret=None, return_poly=True)
        self.assertTrue(isinstance(poly, Polygon))
        _, poly = poly_intersection(
            poly2, poly3, invalid_ret=1, return_poly=True)
        self.assertTrue(poly is None)

    def test_poly_union(self):

        # test unsupported type
        with self.assertRaises(AssertionError):
            poly_union(0, 1)

        # test non-overlapping polygons

        points = [0, 0, 0, 1, 1, 1, 1, 0]
        points1 = [2, 2, 2, 3, 3, 3, 3, 2]
        points2 = [0, 0, 0, 0, 0, 0, 0, 0]  # Invalid polygon
        points3 = [0, 0, 0, 1, 1, 0, 1, 1]  # Self-intersected polygon
        points4 = [0.5, 0.5, 1, 0, 1, 1, 0.5, 0.5]
        poly = poly2shapely(points)
        poly1 = poly2shapely(points1)
        poly2 = poly2shapely(points2)
        poly3 = poly2shapely(points3)
        poly4 = poly2shapely(points4)

        assert poly_union(poly, poly1) == 2

        # test overlapping polygons
        assert poly_union(poly, poly) == 1

        # test invalid polygons
        self.assertEqual(poly_union(poly2, poly2), 0)
        self.assertEqual(poly_union(poly3, poly3, invalid_ret=1), 1)

        # The return value depends on the implementation of the package
        self.assertEqual(poly_union(poly3, poly3, invalid_ret=None), 0.25)
        self.assertEqual(poly_union(poly2, poly3), 0.25)
        self.assertEqual(poly_union(poly3, poly4), 0.5)

        # test poly return
        _, poly = poly_union(poly, poly1, return_poly=True)
        self.assertTrue(isinstance(poly, MultiPolygon))
        _, poly = poly_union(poly3, poly3, return_poly=True)
        self.assertTrue(isinstance(poly, Polygon))
        _, poly = poly_union(poly2, poly3, invalid_ret=0, return_poly=True)
        self.assertTrue(poly is None)

    def test_poly_iou(self):
        # test unsupported type
        with self.assertRaises(AssertionError):
            poly_iou([1], [2])

        points = [0, 0, 0, 1, 1, 1, 1, 0]
        points1 = [10, 20, 30, 40, 50, 60, 70, 80]
        points2 = [0, 0, 0, 0, 0, 0, 0, 0]  # Invalid polygon
        points3 = [0, 0, 0, 1, 1, 0, 1, 1]  # Self-intersected polygon

        poly = poly2shapely(points)
        poly1 = poly2shapely(points1)
        poly2 = poly2shapely(points2)
        poly3 = poly2shapely(points3)

        self.assertEqual(poly_iou(poly, poly1), 0)

        # test overlapping polygons
        self.assertEqual(poly_iou(poly, poly), 1)

        # test invalid polygons
        self.assertEqual(poly_iou(poly2, poly2), 0)
        self.assertEqual(poly_iou(poly3, poly3, zero_division=1), 1)
        self.assertEqual(poly_iou(poly2, poly3), 0)

    def test_offset_polygon(self):
        # usual case
        polygons = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.float32)
        expanded_polygon = offset_polygon(polygons, 1)
        self.assertTrue(
            poly2shapely(expanded_polygon).equals(
                poly2shapely(
                    np.array(
                        [2, 0, 2, 1, 1, 2, 0, 2, -1, 1, -1, 0, 0, -1, 1,
                         -1]))))

        # Overshrunk polygon doesn't exist
        shrunk_polygon = offset_polygon(polygons, -10)
        self.assertEqual(len(shrunk_polygon), 0)

        # When polygon is shrunk into two polygons, it is regarded as invalid
        # and an empty array is returned.
        polygons = np.array([0, 0, 0, 3, 1, 2, 2, 3, 2, 0, 1, 1],
                            dtype=np.float32)
        shrunk = offset_polygon(polygons, -1)
        self.assertEqual(len(shrunk), 0)

    def test_boundary_iou(self):
        points = [0, 0, 0, 1, 1, 1, 1, 0]
        points1 = [10, 20, 30, 40, 50, 60, 70, 80]
        points2 = [0, 0, 0, 0, 0, 0, 0, 0]  # Invalid polygon
        points3 = [0, 0, 0, 1, 1, 0, 1, 1]  # Self-intersected polygon

        self.assertEqual(boundary_iou(points, points1), 0)

        # test overlapping boundaries
        self.assertEqual(boundary_iou(points, points), 1)

        # test invalid boundaries
        self.assertEqual(boundary_iou(points2, points2), 0)
        self.assertEqual(boundary_iou(points3, points3, zero_division=1), 1)
        self.assertEqual(boundary_iou(points2, points3), 0)
