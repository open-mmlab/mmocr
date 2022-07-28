# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from unittest import TestCase

import numpy as np

from mmocr.datasets.transforms import (LoadImageFromFile, LoadImageFromLMDB,
                                       LoadKIEAnnotations, LoadOCRAnnotations)


class TestLoadImageFromFile(TestCase):

    def test_load_img(self):
        data_prefix = osp.join(
            osp.dirname(__file__), '../../data/rec_toy_dataset/imgs/')

        results = dict(img_path=osp.join(data_prefix, '1036169.jpg'))
        transform = LoadImageFromFile(min_size=0)
        results = transform(copy.deepcopy(results))
        self.assertEquals(results['img_path'],
                          osp.join(data_prefix, '1036169.jpg'))
        self.assertEquals(results['img'].shape, (25, 119, 3))
        self.assertEquals(results['img'].dtype, np.uint8)
        self.assertEquals(results['img_shape'], (25, 119))
        self.assertEquals(results['ori_shape'], (25, 119))
        self.assertEquals(
            repr(transform),
            ('LoadImageFromFile(ignore_empty=False, min_size=0, '
             "to_float32=False, color_type='color', imdecode_backend='cv2', "
             "file_client_args={'backend': 'disk'})"))

        transform = LoadImageFromFile(min_size=26)
        results = transform(copy.deepcopy(results))
        self.assertIsNone(results)


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


class TestLoadKIEAnnotations(TestCase):

    def setUp(self):
        self.results = {
            'bboxes': np.random.rand(2, 4).astype(np.float32),
            'bbox_labels': np.random.randint(0, 10, (2, )),
            'edge_labels': np.random.randint(0, 10, (2, 2)),
            'texts': ['text1', 'text2'],
            'ori_shape': (288, 512)
        }
        self.results = {
            'img_shape': (288, 512),
            'ori_shape': (288, 512),
            'instances': [{
                'bbox': [0, 0, 10, 20],
                'bbox_label': 1,
                'edge_label': 1,
                'text': 'tmp1',
            }, {
                'bbox': [10, 10, 110, 120],
                'bbox_label': 2,
                'edge_label': 1,
                'text': 'tmp2',
            }]
        }
        self.load = LoadKIEAnnotations()

    def test_transform(self):
        results = self.load(copy.deepcopy(self.results))

        self.assertIn('gt_bboxes', results)
        self.assertIsInstance(results['gt_bboxes'], np.ndarray)
        self.assertEqual(results['gt_bboxes'].shape, (2, 4))
        self.assertEqual(results['gt_bboxes'].dtype, np.float32)

        self.assertIn('gt_bboxes_labels', results)
        self.assertIsInstance(results['gt_bboxes_labels'], np.ndarray)
        self.assertEqual(results['gt_bboxes_labels'].shape, (2, ))
        self.assertEqual(results['gt_bboxes_labels'].dtype, np.int64)

        self.assertIn('gt_edges_labels', results)
        self.assertIsInstance(results['gt_edges_labels'], np.ndarray)
        self.assertEqual(results['gt_edges_labels'].shape, (2, 2))
        self.assertEqual(results['gt_edges_labels'].dtype, np.int64)

        self.assertIn('ori_shape', results)
        self.assertEqual(results['ori_shape'], (288, 512))

        load = LoadKIEAnnotations(key_node_idx=1, value_node_idx=2)
        results = load(copy.deepcopy(self.results))
        self.assertEqual(results['gt_edges_labels'][0, 1], 1)
        self.assertEqual(results['gt_edges_labels'][1, 0], -1)

    def test_repr(self):
        self.assertEqual(
            repr(self.load),
            'LoadKIEAnnotations(with_bbox=True, with_label=True, '
            'with_text=True)')


class TestLoadImageFromLMDB(TestCase):

    def setUp(self):
        img_key = 'image-%09d' % 1
        self.results1 = {
            'img_path': f'tests/data/rec_toy_dataset/imgs.lmdb/{img_key}'
        }

        img_key = 'image-%09d' % 100
        self.results2 = {
            'img_path': f'tests/data/rec_toy_dataset/imgs.lmdb/{img_key}'
        }

    def test_transform(self):
        transform = LoadImageFromLMDB()
        results = transform(copy.deepcopy(self.results1))
        self.assertIn('img', results)
        self.assertIsInstance(results['img'], np.ndarray)
        self.assertEqual(results['img'].shape[:2], results['img_shape'])
        self.assertEqual(results['ori_shape'], results['img_shape'])

    def test_invalid_key(self):
        transform = LoadImageFromLMDB()
        results = transform(copy.deepcopy(self.results2))
        self.assertEqual(results, None)

    def test_to_float32(self):
        transform = LoadImageFromLMDB(to_float32=True)
        results = transform(copy.deepcopy(self.results1))
        self.assertIn('img', results)
        self.assertIsInstance(results['img'], np.ndarray)
        self.assertTrue(results['img'].dtype, np.float32)
        self.assertEqual(results['img'].shape[:2], results['img_shape'])
        self.assertEqual(results['ori_shape'], results['img_shape'])

    def test_repr(self):
        transform = LoadImageFromLMDB()
        assert repr(transform) == (
            'LoadImageFromLMDB(ignore_empty=False, '
            "to_float32=False, color_type='color', "
            "imdecode_backend='cv2', "
            "file_client_args={'backend': 'lmdb', 'db_path': ''})")
