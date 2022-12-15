# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np

from mmocr.visualization import BaseLocalVisualizer


class TestBaseLocalVisualizer(TestCase):

    def test_get_labels_image(self):
        labels = ['a', 'b', 'c']
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        bboxes = np.array([[0, 0, 10, 10], [10, 10, 20, 20], [20, 20, 30, 30]])
        labels_image = BaseLocalVisualizer().get_labels_image(
            image,
            labels,
            bboxes=bboxes,
            auto_font_size=True,
            colors=['r', 'r', 'r', 'r'])
        self.assertEqual(labels_image.shape, (40, 40, 3))

    def test_get_polygons_image(self):
        polygons = [np.array([0, 0, 10, 10, 20, 20, 30, 30]).reshape(-1, 2)]
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        polygons_image = BaseLocalVisualizer().get_polygons_image(
            image, polygons, colors=['r', 'r', 'r', 'r'])
        self.assertEqual(polygons_image.shape, (40, 40, 3))

        polygons_image = BaseLocalVisualizer().get_polygons_image(
            image, polygons, colors=['r', 'r', 'r', 'r'], filling=True)
        self.assertEqual(polygons_image.shape, (40, 40, 3))

    def test_get_bboxes_image(self):
        bboxes = np.array([[0, 0, 10, 10], [10, 10, 20, 20], [20, 20, 30, 30]])
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        bboxes_image = BaseLocalVisualizer().get_bboxes_image(
            image, bboxes, colors=['r', 'r', 'r', 'r'])
        self.assertEqual(bboxes_image.shape, (40, 40, 3))

        bboxes_image = BaseLocalVisualizer().get_bboxes_image(
            image, bboxes, colors=['r', 'r', 'r', 'r'], filling=True)
        self.assertEqual(bboxes_image.shape, (40, 40, 3))

    def test_cat_images(self):
        image1 = np.zeros((40, 40, 3), dtype=np.uint8)
        image2 = np.zeros((40, 40, 3), dtype=np.uint8)
        image = BaseLocalVisualizer()._cat_image([image1, image2], axis=1)
        self.assertEqual(image.shape, (40, 80, 3))

        image = BaseLocalVisualizer()._cat_image([], axis=0)
        self.assertIsNone(image)

        image = BaseLocalVisualizer()._cat_image([image1, None], axis=0)
        self.assertEqual(image.shape, (40, 40, 3))
