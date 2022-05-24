# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

import numpy as np

from mmocr.datasets.pipelines import LoadOCRAnnotations


class TestLoadOCRAnnotations(TestCase):

    def setUp(self):
        self.results = {
            'height':
            288,
            'width':
            512,
            'instances': [{
                'bbox': [0, 0, 10, 20],
                'bbox_label': 1,
                'polygon': [0, 0, 0, 20, 10, 20, 10, 0],
                'text': 'tmp1',
                'ignore': False
            }, {
                'bbox': [10, 10, 110, 120],
                'bbox_label': 2,
                'polygon': [10, 10, 10, 120, 110, 120, 110, 10],
                'text': 'tmp2',
                'ignore': False
            }, {
                'bbox': [0, 0, 10, 20],
                'bbox_label': 1,
                'polygon': [0, 0, 0, 20, 10, 20, 10, 0],
                'text': 'tmp3',
                'ignore': True
            }, {
                'bbox': [10, 10, 110, 120],
                'bbox_label': 2,
                'polygon': [10, 10, 10, 120, 110, 120, 110, 10],
                'text': 'tmp4',
                'ignore': True
            }]
        }

    def test_load_polygon(self):
        transform = LoadOCRAnnotations(
            with_bbox=False, with_label=False, with_polygon=True)
        results = transform(copy.deepcopy(self.results))
        self.assertIn('gt_polygons', results)
        self.assertIsInstance(results['gt_polygons'], list)
        self.assertEqual(len(results['gt_polygons']), 4)
        for gt_polygon in results['gt_polygons']:
            self.assertIsInstance(gt_polygon, np.ndarray)
            self.assertEqual(gt_polygon.dtype, np.float32)

        self.assertIn('gt_ignored', results)
        self.assertEqual(results['gt_ignored'].dtype, np.bool_)
        self.assertTrue((results['gt_ignored'],
                         np.array([False, False, True, True], dtype=np.bool_)))

    def test_load_text(self):
        transform = LoadOCRAnnotations(
            with_bbox=False, with_label=False, with_text=True)
        results = transform(copy.deepcopy(self.results))
        self.assertIn('gt_texts', results)
        self.assertListEqual(results['gt_texts'],
                             ['tmp1', 'tmp2', 'tmp3', 'tmp4'])

    def test_repr(self):
        transform = LoadOCRAnnotations(
            with_bbox=True, with_label=True, with_polygon=True, with_text=True)
        self.assertEqual(
            repr(transform),
            ('LoadOCRAnnotations(with_bbox=True, with_label=True, '
             'with_polygon=True, with_text=True, '
             "imdecode_backend='cv2', file_client_args={'backend': 'disk'})"))
