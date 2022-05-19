# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest
import unittest.mock as mock

import numpy as np
from mmcv.transforms import Pad

from mmocr.datasets.pipelines import (PyramidRescale, RandomCrop, RandomRotate,
                                      Resize, TextDetRandomCrop,
                                      TextDetRandomCropFlip)
from mmocr.utils import bbox2poly


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
        self.assertEqual(
            repr(transform),
            ('PyramidRescale(factor = 4, randomize_factor = False, '
             'base_w = 128, base_h = 512)'))


class TestTextDetRandomCropFlip(unittest.TestCase):

    def setUp(self):
        img = np.ones((10, 10, 3))
        img[0, 0, :] = 0
        self.data_info1 = dict(
            img=copy.deepcopy(img),
            gt_polygons=[np.array([0., 0., 0., 10., 10., 10., 10., 0.])],
            img_shape=[10, 10])
        self.data_info2 = dict(
            img=copy.deepcopy(img),
            gt_polygons=[np.array([1., 1., 1., 9., 9., 9., 9., 1.])],
            img_shape=[10, 10])

    def test_init(self):
        # iter_num is int
        transform = TextDetRandomCropFlip(iter_num=1)
        self.assertEqual(transform.iter_num, 1)
        # iter_num is float
        with self.assertRaisesRegex(TypeError,
                                    '`iter_num` should be an integer'):
            transform = TextDetRandomCropFlip(iter_num=1.5)

    @mock.patch('mmocr.datasets.pipelines.processing.np.random.randint')
    def test_transforms(self, mock_sample):
        mock_sample.side_effect = [0, 1, 2]
        transform = TextDetRandomCropFlip(crop_ratio=1.0, iter_num=3)
        results = transform(self.data_info2)
        self.assertTrue(np.allclose(results['img'], self.data_info2['img']))
        self.assertTrue(
            np.allclose(results['gt_polygons'],
                        self.data_info2['gt_polygons']))

    def test_generate_crop_target(self):
        transform = TextDetRandomCropFlip(
            crop_ratio=1.0, iter_num=3, pad_ratio=0.1)
        h, w = self.data_info1['img_shape']
        pad_h = int(h * transform.pad_ratio)
        pad_w = int(w * transform.pad_ratio)
        h_axis, w_axis = transform._generate_crop_target(
            self.data_info1['img'], self.data_info1['gt_polygons'], pad_h,
            pad_w)
        self.assertTrue(np.allclose(h_axis, (0, 11)))
        self.assertTrue(np.allclose(w_axis, (0, 11)))

    def test_repr(self):
        transform = TextDetRandomCropFlip(
            pad_ratio=0.1,
            crop_ratio=0.5,
            iter_num=1,
            min_area_ratio=0.2,
            epsilon=1e-2)
        self.assertEqual(
            repr(transform),
            ('TextDetRandomCropFlip(pad_ratio = 0.1, crop_ratio = 0.5, '
             'iter_num = 1, min_area_ratio = 0.2, epsilon = 0.01)'))


class TestRandomRotate(unittest.TestCase):

    def setUp(self):
        img = np.random.random((5, 5))
        self.data_info1 = dict(img=img.copy())
        self.data_info2 = dict(
            img=np.random.random((30, 30, 3)),
            gt_bboxes=np.array([[10, 10, 20, 20], [5, 5, 10, 10]]))
        self.data_info3 = dict(
            img=np.random.random((30, 30, 3)),
            gt_polygons=[np.array([10., 10., 20., 10., 20., 20., 10., 20.])])

    def test_init(self):
        # max angle is float
        with self.assertRaisesRegex(TypeError,
                                    '`max_angle` should be an integer'):
            RandomRotate(max_angle=16.8)
        # invalid pad value
        with self.assertRaisesRegex(
                ValueError, '`pad_value` should contain three integers'):
            RandomRotate(pad_value=[16.8, 0.1])

    def test_transform(self):
        self._test_recog()
        self._test_bboxes()
        self._test_polygons()

    def _test_recog(self):
        # test random rotate for recognition (image only) input
        transform = RandomRotate(max_angle=10)
        results = transform(copy.deepcopy(self.data_info1))
        self.assertTrue(np.allclose(results['img'], self.data_info1['img']))

    @mock.patch('mmocr.datasets.pipelines.processing.np.random.random_sample')
    def _test_bboxes(self, mock_sample):
        # test random rotate for bboxes
        # returns 1. for random_sample() in _sample_angle(), i.e., angle = 90
        mock_sample.side_effect = [1.]
        transform = RandomRotate(max_angle=90, use_canvas=True)
        results = transform(copy.deepcopy(self.data_info2))
        self.assertTrue(
            np.allclose(results['gt_bboxes'][0], np.array([10, 10, 20, 20])))
        self.assertTrue(
            np.allclose(results['gt_bboxes'][1], np.array([5, 20, 10, 25])))
        self.assertEqual(results['img'].shape, self.data_info2['img'].shape)

    @mock.patch('mmocr.datasets.pipelines.processing.np.random.random_sample')
    def _test_polygons(self, mock_sample):
        # test random rotate for polygons
        # returns 1. for random_sample() in _sample_angle(), i.e., angle = 90
        mock_sample.side_effect = [1.]
        transform = RandomRotate(max_angle=90, use_canvas=True)
        results = transform(copy.deepcopy(self.data_info3))
        self.assertTrue(
            np.allclose(results['gt_polygons'][0],
                        np.array([10., 20., 10., 10., 20., 10., 20., 20.])))
        self.assertEqual(results['img'].shape, self.data_info3['img'].shape)

    def test_repr(self):
        transform = RandomRotate(
            max_angle=10,
            pad_with_fixed_color=False,
            pad_value=(0, 0, 0),
            use_canvas=False)
        self.assertEqual(
            repr(transform),
            ('RandomRotate(max_angle = 10, '
             'pad_with_fixed_color = False, pad_value = (0, 0, 0), '
             'use_canvas = False)'))


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


class TestRandomCrop(unittest.TestCase):

    def setUp(self):
        img = np.zeros((30, 30, 3))
        gt_polygons = [
            np.array([5., 5., 25., 5., 25., 10., 5., 10.]),
            np.array([5., 20., 25., 20., 25., 25., 5., 25.])
        ]
        gt_bboxes = np.array([[5, 5, 25, 10], [5, 20, 25, 25]])
        labels = np.array([0, 1])
        gt_ignored = np.array([True, False], dtype=bool)
        texts = ['text1', 'text2']
        self.data_info = dict(
            img=img,
            gt_polygons=gt_polygons,
            gt_bboxes=gt_bboxes,
            gt_bboxes_labels=labels,
            gt_ignored=gt_ignored,
            gt_texts=texts)

    @mock.patch('mmocr.datasets.pipelines.processing.np.random.randint')
    def test_sample_crop_box(self, mock_randint):
        trans = RandomCrop(min_side_ratio=0.3)
        mock_randint.side_effect = [0, 0, 0, 0, 30, 0, 0, 0, 15]
        crop_box = trans._sample_crop_box((30, 30), self.data_info.copy())
        assert np.allclose(np.array(crop_box), np.array([0, 0, 30, 15]))

    @mock.patch('mmocr.datasets.pipelines.processing.np.random.randint')
    def test_transform(self, mock_randint):
        mock_randint.side_effect = [0, 0, 0, 0, 30, 0, 0, 0, 15]
        trans = RandomCrop(min_side_ratio=0.3)
        polygon_target = np.array([5., 5., 25., 5., 25., 10., 5., 10.])
        bbox_target = np.array([[5., 5., 25., 10.]])
        results = trans(self.data_info)

        self.assertEqual(results['img'].shape, (15, 30, 3))
        self.assertEqual(results['img_shape'], (15, 30))
        self.assertEqual(results['gt_bboxes'].all(), bbox_target.all())
        self.assertEqual(results['gt_bboxes'].shape, (1, 4))
        self.assertTrue(len(results['gt_polygons']) == 1)
        self.assertEqual(results['gt_polygons'][0].all(), polygon_target.all())
        self.assertEqual(results['gt_bboxes_labels'][0], 0)
        self.assertEqual(results['gt_ignored'][0], True)
        self.assertEqual(results['gt_texts'][0], 'text1')

    def test_repr(self):
        transform = RandomCrop(min_side_ratio=0.4)
        print(repr(transform))
        self.assertEqual(repr(transform), ('RandomCrop(min_side_ratio = 0.4)'))


class TestTextDetRandomCrop(unittest.TestCase):

    def setUp(self):
        img = np.zeros((5, 5, 3))
        gt_polygons = [np.array([2, 2, 5, 2, 5, 5, 2, 5])]
        gt_bboxes = np.array([[2, 2, 5, 5]])
        gt_bboxes_labels = np.array([0])
        gt_ignored = np.array([True], dtype=bool)
        self.data_info = dict(
            img=img,
            gt_polygons=gt_polygons,
            gt_bboxes=gt_bboxes,
            gt_bboxes_labels=gt_bboxes_labels,
            gt_ignored=gt_ignored)

    @mock.patch('mmocr.datasets.pipelines.processing.np.random.random_sample')
    @mock.patch('mmocr.datasets.pipelines.processing.np.random.randint')
    def test_sample_offset(self, mock_randint, mock_sample):
        # test target size is bigger than image size
        mock_sample.side_effect = [1]
        trans = TextDetRandomCrop(target_size=(6, 6))
        offset = trans._sample_offset(self.data_info['gt_polygons'],
                                      self.data_info['img'].shape[:2])
        self.assertEqual(offset, (0, 0))

        # test the first bracnh in sample_offset
        mock_sample.side_effect = [0.1]
        mock_randint.side_effect = [1, 1]
        trans = TextDetRandomCrop(target_size=(3, 3))
        offset = trans._sample_offset(self.data_info['gt_polygons'],
                                      self.data_info['img'].shape[:2])
        self.assertEqual(offset, (1, 1))

        # test the second branch in sample_offset
        mock_sample.side_effect = [1]
        mock_randint.side_effect = [1, 2]
        trans = TextDetRandomCrop(target_size=(3, 3))
        offset = trans._sample_offset(self.data_info['gt_polygons'],
                                      self.data_info['img'].shape[:2])
        self.assertEqual(offset, (1, 2))

        mock_sample.side_effect = [1]
        mock_randint.side_effect = [1, 2]
        trans = TextDetRandomCrop(target_size=(5, 5))
        offset = trans._sample_offset(self.data_info['gt_polygons'],
                                      self.data_info['img'].shape[:2])
        self.assertEqual(offset, (0, 0))

    def test_crop_image(self):
        img = self.data_info['img']
        offset = [0, 0]
        target = [6, 6]
        trans = TextDetRandomCrop(target_size=(3, 3))
        crop = trans._crop_img(img, offset, target)
        self.assertEqual(img.shape, crop[0].shape)

        target = [3, 2]
        crop = trans._crop_img(img, offset, target)
        self.assertEqual(
            np.array([[0, 0], [0, 0], [0, 0]]).all(), crop[0].all())
        self.assertEqual(crop[1].all(), np.array([0, 0, 2, 3]).all())

    def test_crop_bboxes(self):
        trans = TextDetRandomCrop(target_size=(3, 3))
        crop_box = np.array([2, 3, 5, 5])
        bboxes = np.array([[2, 3, 4, 4], [0, 0, 1, 1], [1, 2, 4, 4],
                           [0, 0, 10, 10]])
        kept_bboxes, kept_idx = trans._crop_bboxes(bboxes, crop_box)
        self.assertEqual(
            kept_bboxes.all(),
            np.array([[0, 0, 2, 1], [0, 0, 2, 1], [0, 0, 3, 2]]).all())
        self.assertEqual(kept_idx, [0, 2, 3])
        self.assertEqual(kept_bboxes.shape, (3, 4))

        bboxes = np.array([[10, 10, 11, 11], [0, 0, 1, 1]])
        kept_bboxes, kept_idx = trans._crop_bboxes(bboxes, crop_box)
        self.assertEqual(kept_bboxes.size, 0)
        self.assertEqual(kept_bboxes.shape, (0, 4))
        self.assertEqual(len(kept_idx), 0)

    def test_crop_polygons(self):
        trans = TextDetRandomCrop(target_size=(3, 3))
        crop_box = np.array([2, 3, 5, 5])
        polygons = [
            bbox2poly([2, 3, 4, 4]),
            bbox2poly([0, 0, 1, 1]),
            bbox2poly([1, 2, 4, 4]),
            bbox2poly([0, 0, 10, 10])
        ]
        kept_polygons, kept_idx = trans._crop_polygons(polygons, crop_box)
        target_polygons = [
            bbox2poly([0, 0, 2, 1]),
            bbox2poly([0, 0, 2, 1]),
            bbox2poly([0, 0, 3, 2]),
        ]
        self.assertEqual(len(kept_polygons), 3)
        self.assertEqual(kept_idx, [0, 2, 3])
        self.assertEqual(target_polygons[0].all(), kept_polygons[0].all())
        self.assertEqual(target_polygons[1].all(), kept_polygons[1].all())
        self.assertEqual(target_polygons[2].all(), kept_polygons[2].all())

    @mock.patch('mmocr.datasets.pipelines.processing.np.random.random_sample')
    @mock.patch('mmocr.datasets.pipelines.processing.np.random.randint')
    def test_transform(self, mock_randint, mock_sample):
        mock_sample.side_effect = [0.1]
        mock_randint.side_effect = [1, 1]
        trans = TextDetRandomCrop(target_size=(3, 3))
        results = trans(self.data_info)
        box_target = np.array([1, 1, 3, 3])
        polygon_target = np.array([1, 1, 3, 1, 3, 3, 1, 3])
        self.assertEqual(results['img'].shape, (3, 3, 3))
        self.assertEqual(results['img_shape'], (3, 3))
        self.assertEqual(box_target.all(), results['gt_bboxes'].all())
        self.assertEqual(polygon_target.all(), results['gt_polygons'][0].all())
        self.assertEqual(results['gt_bboxes_labels'].all(),
                         np.array([0]).all())
        self.assertEqual(results['gt_ignored'][0], True)

    def test_repr(self):
        transform = TextDetRandomCrop(
            target_size=(512, 512), positive_sample_ratio=0.4)
        print(repr(transform))
        self.assertEqual(
            repr(transform), ('TextDetRandomCrop(target_size = (512, 512), '
                              'positive_sample_ratio = 0.4)'))


class TestEastRandomCrop(unittest.TestCase):

    def setUp(self):
        img = np.ones((30, 30, 3))
        gt_polygons = [
            np.array([5., 5., 25., 5., 25., 10., 5., 10.]),
            np.array([5., 20., 25., 20., 25., 25., 5., 25.])
        ]
        gt_bboxes = np.array([[5, 5, 25, 10], [5, 20, 25, 25]])
        labels = np.array([0, 1])
        gt_ignored = np.array([True, False], dtype=bool)
        texts = ['text1', 'text2']
        self.data_info = dict(
            img=img,
            gt_polygons=gt_polygons,
            gt_bboxes=gt_bboxes,
            gt_bboxes_labels=labels,
            gt_ignored=gt_ignored,
            gt_texts=texts)

    @mock.patch('mmocr.datasets.pipelines.processing.np.random.randint')
    def test_east_random_crop(self, mock_randint):

        # test randomcrop
        randcrop = RandomCrop(min_side_ratio=0.5)
        mock_randint.side_effect = [0, 0, 0, 0, 30, 0, 0, 0, 15]
        crop_results = randcrop(self.data_info)
        polygon_target = np.array([5., 5., 25., 5., 25., 10., 5., 10.])
        bbox_target = np.array([[5., 5., 25., 10.]])
        self.assertEqual(crop_results['img'].shape, (15, 30, 3))
        self.assertEqual(crop_results['img_shape'], (15, 30))
        self.assertEqual(crop_results['gt_bboxes'].all(), bbox_target.all())
        self.assertEqual(crop_results['gt_bboxes'].shape, (1, 4))
        self.assertTrue(len(crop_results['gt_polygons']) == 1)
        self.assertEqual(crop_results['gt_polygons'][0].all(),
                         polygon_target.all())
        self.assertEqual(crop_results['gt_bboxes_labels'][0], 0)
        self.assertEqual(crop_results['gt_ignored'][0], True)
        self.assertEqual(crop_results['gt_texts'][0], 'text1')

        # test resize
        resize = Resize(scale=(30, 30), keep_ratio=True)
        resize_results = resize(crop_results)
        self.assertEqual(resize_results['img'].shape, (15, 30, 3))
        self.assertEqual(crop_results['img_shape'], (15, 30))
        self.assertEqual(crop_results['scale'], (30, 15))
        self.assertEqual(crop_results['scale_factor'], (1., 1.))
        self.assertEqual(crop_results['keep_ratio'], True)

        # test pad
        pad = Pad(size=(30, 30))
        pad_results = pad(resize_results)
        self.assertEqual(pad_results['img'].shape, (30, 30, 3))
        self.assertEqual(pad_results['pad_shape'], (30, 30, 3))
        self.assertEqual(pad_results['img'].sum(), 15 * 30 * 3)
