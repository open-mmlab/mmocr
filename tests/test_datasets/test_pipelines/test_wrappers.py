# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest
from typing import Dict, List, Optional

import numpy as np
from shapely.geometry import Polygon

from mmocr.datasets.pipelines import ImgAug


class TestImgAug(unittest.TestCase):

    def test_init(self):
        with self.assertRaises(AssertionError):
            ImgAug(args=[])
        with self.assertRaises(AssertionError):
            ImgAug(args=['test'])

    def _create_dummy_data(self):
        img = np.random.rand(50, 50, 3)
        poly = np.array([[[0, 0, 50, 0, 50, 50, 0, 50]],
                         [[20, 20, 50, 20, 50, 50, 20, 50]]])
        box = np.array([[0, 0, 50, 50], [20, 20, 50, 50]])
        ignores = np.array([False, True], dtype=bool)
        texts = ['text1', 'text2']
        return dict(
            img=img,
            img_shape=img.shape,
            gt_polygons=poly,
            gt_bboxes=box,
            gt_ignores=ignores,
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
                            ignore_targets: np.ndarray,
                            text_targets: Optional[List[str]] = None) -> None:
        self.assertPolyEqual(poly_targets, results['gt_polygons'])
        self.assertTrue(np.allclose(bbox_targets, results['gt_bboxes']))
        self.assertTrue(np.allclose(ignore_targets, results['gt_ignores']))
        self.assertEqual(text_targets, results['gt_texts'])
        self.assertEqual(results['img_shape'], results['img'].shape)

    def test_transform(self):

        imgaug_transform = ImgAug()
        results = self._create_dummy_data()
        origin_results = copy.deepcopy(results)
        results = imgaug_transform(results)
        self.assert_result_equal(results, origin_results['gt_polygons'],
                                 origin_results['gt_bboxes'],
                                 origin_results['gt_ignores'],
                                 origin_results['gt_texts'])

        args = [dict(cls='Affine', translate_px=dict(x=-10, y=-10))]
        imgaug_transform = ImgAug(args)
        results = self._create_dummy_data()
        results = imgaug_transform(results)

        poly_target = [
            np.array([0, 0, 40, 0, 40, 40, 0, 40]),
            np.array([10, 10, 40, 10, 40, 40, 10, 40])
        ]
        box_target = np.array([[0, 0, 40, 40], [10, 10, 40, 40]])
        ignores = np.array([False, True], dtype=bool)
        texts = ['text1', 'text2']
        self.assert_result_equal(results, poly_target, box_target, ignores,
                                 texts)

        args = [
            dict(cls='Affine', translate_px=dict(x=30, y=30)), ['Fliplr', 1]
        ]
        poly_target = [np.array([0, 30, 20, 30, 20, 50, 0, 50])]
        box_target = np.array([[0, 30, 20, 50]])
        ignores = np.array([False], dtype=bool)
        texts = ['text1']
        imgaug_transform = ImgAug(args)
        results = self._create_dummy_data()
        results = imgaug_transform(results)
        self.assert_result_equal(results, poly_target, box_target, ignores,
                                 texts)

        results = self._create_dummy_data()
        del results['gt_texts']
        results = imgaug_transform(results)
        self.assertNotIn('gt_texts', results)

    def test_repr(self):
        args = [['Resize', [0.5, 3.0]], ['Fliplr', 0.5]]
        transform = ImgAug(args)
        print(repr(transform))
        self.assertEqual(
            repr(transform),
            ("ImgAug(args = [['Resize', [0.5, 3.0]], ['Fliplr', 0.5]])"))
