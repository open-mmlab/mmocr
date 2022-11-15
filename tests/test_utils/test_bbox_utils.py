# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch

from mmocr.utils import (bbox2poly, bbox_center_distance, bbox_diag_distance,
                         bezier2polygon, is_on_same_line,
                         stitch_boxes_into_lines)
from mmocr.utils.bbox_utils import bbox_jitter


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
        self.assertTrue(
            np.array_equal(bbox2poly(self.box_list, mode='xywh'), self.gt))
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


class TestBboxJitter(unittest.TestCase):

    def test_bbox_jitter(self):
        dummy_points_x = [20, 120, 120, 20]
        dummy_points_y = [20, 20, 40, 40]

        kwargs = dict(jitter_ratio_x=0.0, jitter_ratio_y=0.0)

        with self.assertRaises(AssertionError):
            bbox_jitter([], dummy_points_y)
        with self.assertRaises(AssertionError):
            bbox_jitter(dummy_points_x, [])
        with self.assertRaises(AssertionError):
            bbox_jitter(dummy_points_x, dummy_points_y, jitter_ratio_x=1.)
        with self.assertRaises(AssertionError):
            bbox_jitter(dummy_points_x, dummy_points_y, jitter_ratio_y=1.)

        bbox_jitter(dummy_points_x, dummy_points_y, **kwargs)

        assert np.allclose(dummy_points_x, [20, 120, 120, 20])
        assert np.allclose(dummy_points_y, [20, 20, 40, 40])


class TestIsOnSameLine(unittest.TestCase):

    def test_box_on_line(self):
        # regular boxes
        box1 = [0, 0, 1, 0, 1, 1, 0, 1]
        box2 = [2, 0.5, 3, 0.5, 3, 1.5, 2, 1.5]
        box3 = [4, 0.8, 5, 0.8, 5, 1.8, 4, 1.8]
        self.assertTrue(is_on_same_line(box1, box2, 0.5))
        self.assertFalse(is_on_same_line(box1, box3, 0.5))

        # irregular box4
        box4 = [0, 0, 1, 1, 1, 2, 0, 1]
        box5 = [2, 1.5, 3, 1.5, 3, 2.5, 2, 2.5]
        box6 = [2, 1.6, 3, 1.6, 3, 2.6, 2, 2.6]
        self.assertTrue(is_on_same_line(box4, box5, 0.5))
        self.assertFalse(is_on_same_line(box4, box6, 0.5))


class TestStitchBoxesIntoLines(unittest.TestCase):

    def test_stitch_boxes_into_lines(self):
        boxes = [  # regular boxes
            [0, 0, 1, 0, 1, 1, 0, 1],
            [2, 0.5, 3, 0.5, 3, 1.5, 2, 1.5],
            [3, 1.2, 4, 1.2, 4, 2.2, 3, 2.2],
            [5, 0.5, 6, 0.5, 6, 1.5, 5, 1.5],
            # irregular box
            [6, 1.5, 7, 1.25, 7, 1.75, 6, 1.75]
        ]
        raw_input = [{
            'box': boxes[i],
            'text': str(i)
        } for i in range(len(boxes))]
        result = stitch_boxes_into_lines(raw_input, 1, 0.5)
        # Final lines: [0, 1], [2], [3, 4]
        # box 0, 1, 3, 4 are on the same line but box 3 is 2 pixels away from
        # box 1
        # box 3 and 4 are on the same line since the length of overlapping part
        # >= 0.5 * the y-axis length of box 5
        expected_result = [{
            'box': [0, 0, 3, 0, 3, 1.5, 0, 1.5],
            'text': '0 1'
        }, {
            'box': [3, 1.2, 4, 1.2, 4, 2.2, 3, 2.2],
            'text': '2'
        }, {
            'box': [5, 0.5, 7, 0.5, 7, 1.75, 5, 1.75],
            'text': '3 4'
        }]
        result.sort(key=lambda x: x['box'][0])
        expected_result.sort(key=lambda x: x['box'][0])
        self.assertEqual(result, expected_result)
