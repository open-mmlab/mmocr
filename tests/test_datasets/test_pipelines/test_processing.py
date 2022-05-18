# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest

import numpy as np

from mmocr.datasets.pipelines import PyramidRescale, Resize


class TestPyramidRescale(unittest.TestCase):

    def setUp(self):
        self.data_info = dict(img=np.random.random((128, 100, 3)))

    def test_init(self):
        # factor is int
        transform = PyramidRescale(factor=4, randomize_factor=False)
        self.assertEqual(transform.factor, 4)
        # factor is float
        with self.assertRaisesRegex(TypeError,
                                    '`factor` should be an integer'):
            PyramidRescale(factor=4.0)
        # invalid base_shape
        with self.assertRaisesRegex(TypeError,
                                    '`base_shape` should be a list or tuple'):
            PyramidRescale(base_shape=128)
        with self.assertRaisesRegex(
                ValueError, '`base_shape` should contain two integers'):
            PyramidRescale(base_shape=(128, ))
        with self.assertRaisesRegex(
                ValueError, '`base_shape` should contain two integers'):
            PyramidRescale(base_shape=(128.0, 2.0))
        # invalid randomize_factor
        with self.assertRaisesRegex(TypeError,
                                    '`randomize_factor` should be a bool'):
            PyramidRescale(randomize_factor=None)

    def test_transform(self):
        # test if the rescale keeps the original size
        transform = PyramidRescale()
        results = transform(copy.deepcopy(self.data_info))
        self.assertEqual(results['img'].shape, (128, 100, 3))
        # test factor = 0
        transform = PyramidRescale(factor=0, randomize_factor=False)
        results = transform(copy.deepcopy(self.data_info))
        self.assertTrue(np.all(results['img'] == self.data_info['img']))

    def test_repr(self):
        transform = PyramidRescale(
            factor=4, base_shape=(128, 512), randomize_factor=False)
        print(repr(transform))
        self.assertEqual(
            repr(transform),
            ('PyramidRescale(factor = 4, randomize_factor = False, '
             'base_w = 128, base_h = 512)'))


class TestResize(unittest.TestCase):

    def setUp(self):
        self.data_info1 = dict(
            img=np.random.random((600, 800, 3)),
            gt_bboxes=np.array([[0, 0, 60, 100]]),
            gt_polygons=[np.array([0, 0, 200, 0, 200, 100, 0, 100])])
        self.data_info2 = dict(
            img=np.random.random((200, 300, 3)),
            gt_bboxes=np.array([[0, 0, 400, 600]]),
            gt_polygons=[np.array([0, 0, 400, 0, 400, 400, 0, 400])])
        self.data_info3 = dict(
            img=np.random.random((200, 300, 3)),
            gt_bboxes=np.array([[400, 400, 600, 600]]),
            gt_polygons=[np.array([400, 400, 500, 400, 500, 600, 400, 600])])

    def test_resize(self):
        # test keep_ratio is True
        transform = Resize(scale=(400, 400), keep_ratio=True)
        results = transform(copy.deepcopy(self.data_info1.copy()))
        self.assertEqual(results['img'].shape[:2], (300, 400))
        self.assertEqual(results['img_shape'], (300, 400))
        self.assertEqual(results['scale'], (400, 300))
        self.assertEqual(results['scale_factor'], (400 / 800, 300 / 600))
        self.assertEqual(results['keep_ratio'], True)

        # test keep_ratio is False
        transform = Resize(scale=(400, 400))
        results = transform(copy.deepcopy(self.data_info1.copy()))
        self.assertEqual(results['img'].shape[:2], (400, 400))
        self.assertEqual(results['img_shape'], (400, 400))
        self.assertEqual(results['scale'], (400, 400))
        self.assertEqual(results['scale_factor'], (400 / 800, 400 / 600))
        self.assertEqual(results['keep_ratio'], False)

        # test resize_bboxes/polygons
        transform = Resize(scale_factor=(1.5, 2))
        results = transform(copy.deepcopy(self.data_info1.copy()))
        self.assertEqual(results['img'].shape[:2], (1200, 1200))
        self.assertEqual(results['img_shape'], (1200, 1200))
        self.assertEqual(results['scale'], (1200, 1200))
        self.assertEqual(results['scale_factor'], (1.5, 2))
        self.assertEqual(results['keep_ratio'], False)
        self.assertTrue(
            results['gt_bboxes'].all() == np.array([[0, 0, 90, 200]]).all())
        self.assertTrue(results['gt_polygons'][0].all() == np.array(
            [0, 0, 300, 0, 300, 200, 0, 200]).all())

        # test clip_object_border = False
        transform = Resize(scale=(150, 100), clip_object_border=False)
        results = transform(self.data_info2.copy())
        self.assertEqual(results['img'].shape[:2], (100, 150))
        self.assertEqual(results['img_shape'], (100, 150))
        self.assertEqual(results['scale'], (150, 100))
        self.assertEqual(results['scale_factor'], (150. / 300., 100. / 200.))
        self.assertEqual(results['keep_ratio'], False)
        self.assertTrue(
            results['gt_bboxes'].all() == np.array([0, 0, 200, 300]).all())
        self.assertTrue(results['gt_polygons'][0].all() == np.array(
            [0, 0, 200, 0, 200, 200, 0, 200]).all())

        # test clip_object_border = True
        transform = Resize(scale=(150, 100), clip_object_border=True)
        results = transform(self.data_info2.copy())
        self.assertEqual(results['img'].shape[:2], (100, 150))
        self.assertEqual(results['img_shape'], (100, 150))
        self.assertEqual(results['scale'], (150, 100))
        self.assertEqual(results['scale_factor'], (150. / 300., 100. / 200.))
        self.assertEqual(results['keep_ratio'], False)
        self.assertTrue(
            results['gt_bboxes'].all() == np.array([0, 0, 150, 100]).all())
        self.assertTrue(results['gt_polygons'][0].shape == (8, ))
        self.assertTrue(results['gt_polygons'][0].all() == np.array(
            [0, 0, 150, 0, 150, 100, 0, 100]).all())

        # test clip_object_border = True and polygon outside image
        transform = Resize(scale=(150, 100), clip_object_border=True)
        results = transform(self.data_info3)
        self.assertEqual(results['img'].shape[:2], (100, 150))
        self.assertEqual(results['img_shape'], (100, 150))
        self.assertEqual(results['scale'], (150, 100))
        self.assertEqual(results['scale_factor'], (150. / 300., 100. / 200.))
        self.assertEqual(results['keep_ratio'], False)
        self.assertEqual(results['gt_polygons'][0].all(),
                         np.array([0., 0., 0., 0., 0., 0., 0., 0.]).all())
        self.assertEqual(results['gt_bboxes'].all(),
                         np.array([[150., 100., 150., 100.]]).all())

    def test_repr(self):
        transform = Resize(scale=(2000, 2000), keep_ratio=True)
        self.assertEqual(
            repr(transform), ('Resize(scale=(2000, 2000), '
                              'scale_factor=None, keep_ratio=True, '
                              'clip_object_border=True), backend=cv2), '
                              'interpolation=bilinear)'))
