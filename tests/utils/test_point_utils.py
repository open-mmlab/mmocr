# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch

from mmocr.utils import point_distance, points_center


class TestPointDistance(unittest.TestCase):

    def setUp(self) -> None:
        self.p1_list = [1, 2]
        self.p2_list = [2, 2]
        self.p1_array = np.array([1, 2])
        self.p2_array = np.array([2, 2])
        self.p1_tensor = torch.Tensor([1, 2])
        self.p2_tensor = torch.Tensor([2, 2])
        self.invalid_p = [1, 2, 3]

    def test_point_distance(self):
        # list
        self.assertEqual(point_distance(self.p1_list, self.p2_list), 1)
        self.assertEqual(point_distance(self.p1_list, self.p1_list), 0)
        # array
        self.assertEqual(point_distance(self.p1_array, self.p2_array), 1)
        self.assertEqual(point_distance(self.p1_array, self.p1_array), 0)
        # tensor
        self.assertEqual(point_distance(self.p1_tensor, self.p2_tensor), 1)
        self.assertEqual(point_distance(self.p1_tensor, self.p1_tensor), 0)
        with self.assertRaises(AssertionError):
            point_distance(self.invalid_p, self.invalid_p)


class TestPointCenter(unittest.TestCase):

    def setUp(self) -> None:
        self.point_list = [1, 2, 3, 4]
        self.point_nparray = np.array([1, 2, 3, 4])
        self.point_tensor = torch.Tensor([1, 2, 3, 4])
        self.incorrect_input = [1, 3, 4]
        self.gt = np.array([2, 3])

    def test_point_center(self):
        # list
        self.assertTrue(
            np.array_equal(points_center(self.point_list), self.gt))
        # array
        self.assertTrue(
            np.array_equal(points_center(self.point_nparray), self.gt))
        # tensor
        self.assertTrue(
            np.array_equal(points_center(self.point_tensor), self.gt))
        with self.assertRaises(AssertionError):
            points_center(self.incorrect_input)
