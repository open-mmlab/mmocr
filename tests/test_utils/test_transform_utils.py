# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest

import numpy as np

from mmocr.utils import remove_pipeline_elements


class TestTransformUtils(unittest.TestCase):

    def test_remove_pipeline_elements(self):
        data = dict(img=np.random.random((30, 40, 3)))
        results = remove_pipeline_elements(copy.deepcopy(data), [0, 1, 2])
        self.assertTrue(np.array_equal(results['img'], data['img']))
        self.assertEqual(len(data), len(results))

        data['gt_polygons'] = [
            np.array([0., 0., 10., 10., 10., 0.]),
            np.array([0., 0., 10., 0., 0., 10.]),
            np.array([0, 10, 0, 10, 1, 2, 3, 4]),
            np.array([0, 10, 0, 10, 10, 0, 0, 10]),
        ]
        data['dummy'] = [
            np.array([0., 0., 10., 10., 10., 0.]),
        ]
        data['gt_ignored'] = np.array([True, True, False, False], dtype=bool)
        data['gt_bboxes_labels'] = np.array([0, 1, 2, 3])
        data['gt_bboxes'] = np.array([[1, 2, 3, 4], [5, 6, 7, 8],
                                      [0, 0, 10, 10], [0, 0, 0, 0]])
        data['gt_texts'] = ['t1', 't2', 't3', 't4']
        keys = [
            'gt_polygons', 'gt_bboxes', 'gt_ignored', 'gt_texts',
            'gt_bboxes_labels'
        ]
        results = remove_pipeline_elements(copy.deepcopy(data), [0, 1, 2])

        for key in keys:
            self.assertTrue(np.array_equal(results[key][0], data[key][3]))
        self.assertTrue(np.array_equal(results['img'], data['img']))
        self.assertTrue(np.array_equal(results['dummy'], data['dummy']))
