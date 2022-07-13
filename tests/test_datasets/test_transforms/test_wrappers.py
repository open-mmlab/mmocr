# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest
from typing import Dict, List, Optional

import numpy as np
from shapely.geometry import Polygon

from mmocr.datasets.transforms import ImgAugWrapper, TorchVisionWrapper


class TestImgAug(unittest.TestCase):

    def test_init(self):
        with self.assertRaises(AssertionError):
            ImgAugWrapper(args=[])
        with self.assertRaises(AssertionError):
            ImgAugWrapper(args=['test'])

    def _create_dummy_data(self):
        img = np.random.rand(50, 50, 3)
        poly = np.array([[[0, 0, 50, 0, 50, 50, 0, 50]],
                         [[20, 20, 50, 20, 50, 50, 20, 50]]])
        box = np.array([[0, 0, 50, 50], [20, 20, 50, 50]])
        # It shall always be 0 in MMOCR, but we assign different labels to
        # dummy instances for testing
        labels = np.array([0, 1], dtype=np.int64)
        ignored = np.array([False, True], dtype=bool)
        texts = ['text1', 'text2']
        return dict(
            img=img,
            img_shape=(50, 50),
            gt_polygons=poly,
            gt_bboxes=box,
            gt_bboxes_labels=labels,
            gt_ignored=ignored,
            gt_texts=texts)

    def assertPolyEqual(self, poly1: List[np.ndarray],
                        poly2: List[np.ndarray]) -> None:
        for p1, p2 in zip(poly1, poly2):
            self.assertTrue(
                Polygon(p1.reshape(-1, 2)).equals(Polygon(p2.reshape(-1, 2))))

    def assert_result_equal(self,
                            results: Dict,
                            poly_targets: List[np.ndarray],
                            bbox_targets: np.ndarray,
                            bbox_label_targets: np.ndarray,
                            ignore_targets: np.ndarray,
                            text_targets: Optional[List[str]] = None) -> None:
        self.assertPolyEqual(poly_targets, results['gt_polygons'])
        self.assertTrue(np.array_equal(bbox_targets, results['gt_bboxes']))
        self.assertTrue(
            np.array_equal(bbox_label_targets, results['gt_bboxes_labels']))
        self.assertTrue(np.array_equal(ignore_targets, results['gt_ignored']))
        self.assertEqual(text_targets, results['gt_texts'])
        self.assertEqual(results['img_shape'],
                         (results['img'].shape[0], results['img'].shape[1]))

    def test_transform(self):

        # Test empty transform
        imgaug_transform = ImgAugWrapper()
        results = self._create_dummy_data()
        origin_results = copy.deepcopy(results)
        results = imgaug_transform(results)
        self.assert_result_equal(results, origin_results['gt_polygons'],
                                 origin_results['gt_bboxes'],
                                 origin_results['gt_bboxes_labels'],
                                 origin_results['gt_ignored'],
                                 origin_results['gt_texts'])

        args = [dict(cls='Affine', translate_px=dict(x=-10, y=-10))]
        imgaug_transform = ImgAugWrapper(args)
        results = self._create_dummy_data()
        results = imgaug_transform(results)

        # Polygons and bboxes are partially outside the image after
        # transformation
        poly_target = [
            np.array([0, 0, 40, 0, 40, 40, 0, 40]),
            np.array([10, 10, 40, 10, 40, 40, 10, 40])
        ]
        box_target = np.array([[0, 0, 40, 40], [10, 10, 40, 40]])
        label_target = np.array([0, 1], dtype=np.int64)
        ignored = np.array([False, True], dtype=bool)
        texts = ['text1', 'text2']
        self.assert_result_equal(results, poly_target, box_target,
                                 label_target, ignored, texts)

        # Some polygons and bboxes are no longer inside the image after
        # transformation
        args = [
            dict(cls='Affine', translate_px=dict(x=30, y=30)), ['Fliplr', 1]
        ]
        poly_target = [np.array([0, 30, 20, 30, 20, 50, 0, 50])]
        box_target = np.array([[0, 30, 20, 50]])
        label_target = np.array([0], dtype=np.int64)
        ignored = np.array([False], dtype=bool)
        texts = ['text1']
        imgaug_transform = ImgAugWrapper(args)
        results = self._create_dummy_data()
        results = imgaug_transform(results)
        self.assert_result_equal(results, poly_target, box_target,
                                 label_target, ignored, texts)

        # All polygons and bboxes are no longer inside the image after
        # transformation

        # When some transforms result in empty polygons
        args = [dict(cls='Affine', translate_px=dict(x=100, y=100))]
        results = self._create_dummy_data()
        invalid_transform = ImgAugWrapper(args)
        results = invalid_transform(results)
        self.assertIsNone(results)

        # Everything should work well without gt_texts
        results = self._create_dummy_data()
        del results['gt_texts']
        results = imgaug_transform(results)
        self.assertNotIn('gt_texts', results)

        # Everything should work well without keys required from text detection
        results = imgaug_transform(
            dict(
                img=np.random.rand(10, 20, 3),
                img_shape=(10, 20),
                gt_texts=['text1', 'text2']))
        self.assertEqual(results['gt_texts'], ['text1', 'text2'])

    def test_repr(self):
        args = [['Resize', [0.5, 3.0]], ['Fliplr', 0.5]]
        transform = ImgAugWrapper(args)
        print(repr(transform))
        self.assertEqual(
            repr(transform),
            ("ImgAugWrapper(args = [['Resize', [0.5, 3.0]], ['Fliplr', 0.5]])"
             ))


class TestTorchVisionWrapper(unittest.TestCase):

    def test_transform(self):
        x = {'img': np.ones((128, 100, 3), dtype=np.uint8)}
        # object not found error
        with self.assertRaises(Exception):
            TorchVisionWrapper(op='NonExist')
        with self.assertRaises(TypeError):
            TorchVisionWrapper()
        f = TorchVisionWrapper('Grayscale')
        with self.assertRaises(AssertionError):
            f({})
        results = f(x)
        assert results['img'].shape == (128, 100)
        assert results['img_shape'] == (128, 100)

    def test_repr(self):
        f = TorchVisionWrapper('Grayscale', num_output_channels=3)
        self.assertEqual(
            repr(f),
            'TorchVisionWrapper(op = Grayscale, num_output_channels = 3)')
