# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch

from mmocr.utils import (bbox2poly, bbox_center_distance, bbox_diag_distance,
                         bezier2polygon)


class TestBbox2poly(unittest.TestCase):

    def setUp(self) -> None:
        self.box_array = np.array([0, 0, 1, 1])
        self.box_list = [0, 0, 1, 1]
        self.box_tensor = torch.tensor([0, 0, 1, 1])
        self.gt = np.array([0, 0, 1, 0, 1, 1, 0, 1])

    def test_bbox2poly(self):
        # test np.array
        self.assertTrue(np.array_equal(bbox2poly(self.box_array), self.gt))
        # test list
        self.assertTrue(np.array_equal(bbox2poly(self.box_list), self.gt))
        # test tensor
        self.assertTrue(np.array_equal(bbox2poly(self.box_tensor), self.gt))


class TestBoxCenterDistance(unittest.TestCase):

    def setUp(self) -> None:
        self.box1_list = [1, 1, 3, 3]
        self.box2_list = [2, 2, 4, 2]
        self.box1_array = np.array([1, 1, 3, 3])
        self.box2_array = np.array([2, 2, 4, 2])
        self.box1_tensor = torch.tensor([1, 1, 3, 3])
        self.box2_tensor = torch.tensor([2, 2, 4, 2])
        self.gt = 1

    def test_box_center_distance(self):
        # test list
        self.assertEqual(
            bbox_center_distance(self.box1_list, self.box2_list), self.gt)
        # test np.array
        self.assertEqual(
            bbox_center_distance(self.box1_array, self.box2_array), self.gt)
        # test tensor
        self.assertEqual(
            bbox_center_distance(self.box1_tensor, self.box2_tensor), self.gt)


class TestBoxDiagDistance(unittest.TestCase):

    def setUp(self) -> None:
        self.box_list1 = [0, 0, 1, 1, 0, 10, -10, 0]
        self.box_array1 = np.array(self.box_list1)
        self.box_tensor1 = torch.tensor(self.box_list1)
        self.gt1 = 10
        self.box_list2 = [0, 0, 1, 1]
        self.box_array2 = np.array(self.box_list2)
        self.box_tensor2 = torch.tensor(self.box_list2)
        self.gt2 = np.sqrt(2)

    def test_bbox_diag_distance(self):
        # quad [x1, y1, x2, y2, x3, y3, x4, y4]
        # list
        self.assertEqual(bbox_diag_distance(self.box_list1), self.gt1)
        # array
        self.assertEqual(bbox_diag_distance(self.box_array1), self.gt1)
        # tensor
        self.assertEqual(bbox_diag_distance(self.box_tensor1), self.gt1)
        # rect [x1, y1, x2, y2]
        # list
        self.assertAlmostEqual(bbox_diag_distance(self.box_list2), self.gt2)
        # array
        self.assertAlmostEqual(bbox_diag_distance(self.box_array2), self.gt2)
        # tensor
        self.assertAlmostEqual(bbox_diag_distance(self.box_tensor2), self.gt2)


class TestBezier2Polygon(unittest.TestCase):

    def setUp(self) -> None:
        self.bezier_points1 = [
            37.0, 249.0, 72.5, 229.55, 95.34, 220.65, 134.0, 216.0, 132.0,
            233.0, 82.11, 240.2, 72.46, 247.16, 38.0, 263.0
        ]
        self.gt1 = np.array([[37.0, 249.0],
                             [42.50420761043885, 246.01570199737577],
                             [47.82291296107305, 243.2012392477038],
                             [52.98102930456334, 240.5511007435486],
                             [58.00346989357049, 238.05977547747486],
                             [62.91514798075522, 235.721752442047],
                             [67.74097681877824, 233.53152062982943],
                             [72.50586966030032, 231.48356903338674],
                             [77.23473975798221, 229.57238664528356],
                             [81.95250036448464, 227.79246245808432],
                             [86.68406473246829, 226.13828546435346],
                             [91.45434611459396, 224.60434465665548],
                             [96.28825776352238, 223.18512902755504],
                             [101.21071293191426, 221.87512756961655],
                             [106.24662487243039, 220.6688292754046],
                             [111.42090683773145, 219.5607231374836],
                             [116.75847208047819, 218.5452981484181],
                             [122.28423385333137, 217.6170433007727],
                             [128.02310540895172, 216.77044758711182],
                             [134.0, 216.0], [132.0, 233.0],
                             [124.4475521213005, 234.13617728531858],
                             [117.50700976818779, 235.2763434903047],
                             [111.12146960198277, 236.42847645429362],
                             [105.2340282840064, 237.6005540166205],
                             [99.78778247557953, 238.80055401662054],
                             [94.72582883802303, 240.0364542936288],
                             [89.99126403265781, 241.31623268698053],
                             [85.52718472080478, 242.64786703601104],
                             [81.27668756378483, 244.03933518005545],
                             [77.1828692229188, 245.49861495844874],
                             [73.18882635952762, 247.0336842105263],
                             [69.23765563493221, 248.65252077562326],
                             [65.27245371045342, 250.3631024930748],
                             [61.23631724741216, 252.17340720221605],
                             [57.07234290712931, 254.09141274238226],
                             [52.723627350925796, 256.12509695290856],
                             [48.13326724012247, 258.2824376731302],
                             [43.24435923604024, 260.5714127423822],
                             [38.0, 263.0]])
        self.bezier_points2 = [0, 0, 0, 1, 0, 2, 0, 3, 1, 0, 1, 1, 1, 2, 1, 3]
        self.gt2 = np.array([[0, 0], [0, 1.5], [0, 3], [1, 0], [1, 1.5],
                             [1, 3]])
        self.invalid_input = [0, 1]

    def test_bezier2polygon(self):
        self.assertTrue(
            np.allclose(bezier2polygon(self.bezier_points1), self.gt1))
        with self.assertRaises(AssertionError):
            bezier2polygon(self.bezier_points2, num_sample=-1)
        with self.assertRaises(AssertionError):
            bezier2polygon(self.invalid_input, num_sample=-1)
