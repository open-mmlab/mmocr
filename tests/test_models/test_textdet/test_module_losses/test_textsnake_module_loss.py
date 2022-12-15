# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine.structures import InstanceData

from mmocr.models.textdet.module_losses import TextSnakeModuleLoss
from mmocr.structures import TextDetDataSample


class TestTextSnakeModuleLoss(TestCase):

    def setUp(self) -> None:
        self.loss = TextSnakeModuleLoss()

        self.data_samples = [
            TextDetDataSample(
                metainfo=dict(img_shape=(3, 10)),
                gt_instances=InstanceData(
                    polygons=np.array([
                        [0, 0, 1, 0, 1, 1, 0, 1],
                        [2, 0, 3, 0, 3, 1, 2, 1],
                    ],
                                      dtype=np.float32),
                    ignored=torch.BoolTensor([False, False])))
        ]
        self.preds = torch.rand((1, 5, 3, 10))

    def test_forward(self):
        loss_output = self.loss(self.preds, self.data_samples)
        self.assertTrue(isinstance(loss_output, dict))
        self.assertIn('loss_text', loss_output)
        self.assertIn('loss_center', loss_output)
        self.assertIn('loss_radius', loss_output)
        self.assertIn('loss_sin', loss_output)
        self.assertIn('loss_cos', loss_output)

    def test_find_head_tail(self):
        # for quadrange
        polygon = np.array([[1.0, 1.0], [5.0, 1.0], [5.0, 3.0], [1.0, 3.0]])
        head_inds, tail_inds = self.loss._find_head_tail(polygon, 2.0)
        self.assertTrue(np.allclose(head_inds, [3, 0]))
        self.assertTrue(np.allclose(tail_inds, [1, 2]))
        polygon = np.array([[1.0, 1.0], [1.0, 3.0], [5.0, 3.0], [5.0, 1.0]])
        head_inds, tail_inds = self.loss._find_head_tail(polygon, 2.0)
        self.assertTrue(np.allclose(head_inds, [0, 1]))
        self.assertTrue(np.allclose(tail_inds, [2, 3]))

        # for polygon
        polygon = np.array([[0., 10.], [3., 3.], [10., 0.], [17., 3.],
                            [20., 10.], [15., 10.], [13.5, 6.5], [10., 5.],
                            [6.5, 6.5], [5., 10.]])
        head_inds, tail_inds = self.loss._find_head_tail(polygon, 2.0)
        self.assertTrue(np.allclose(head_inds, [9, 0]))
        self.assertTrue(np.allclose(tail_inds, [4, 5]))

    def test_vector_angle(self):
        v1 = np.array([[-1, 0], [0, 1]])
        v2 = np.array([[1, 0], [0, 1]])
        angles = self.loss.vector_angle(v1, v2)
        self.assertTrue(np.allclose(angles, np.array([np.pi, 0]), atol=1e-3))

    def test_resample_line(self):
        # test resample_line
        line = np.array([[0, 0], [0, 1], [0, 3], [0, 4], [0, 7], [0, 8]])
        resampled_line = self.loss._resample_line(line, 3)
        self.assertEqual(len(resampled_line), 3)
        self.assertTrue(
            np.allclose(resampled_line, np.array([[0, 0], [0, 4], [0, 8]])))
        line = np.array([[0, 0], [0, 0]])
        resampled_line = self.loss._resample_line(line, 4)
        self.assertEqual(len(resampled_line), 4)
        self.assertTrue(
            np.allclose(resampled_line,
                        np.array([[0, 0], [0, 0], [0, 0], [0, 0]])))

    def test_generate_text_region_mask(self):
        img_size = (3, 10)
        text_polys = [
            np.array([0, 0, 1, 0, 1, 1, 0, 1]),
            np.array([2, 0, 3, 0, 3, 1, 2, 1])
        ]
        output = self.loss._generate_text_region_mask(img_size, text_polys)
        target = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.assertTrue(np.allclose(output, target))

    def test_generate_center_mask_attrib_maps(self):
        img_size = (3, 10)
        text_polys = [
            np.array([0, 0, 1, 0, 1, 1, 0, 1]),
            np.array([2, 0, 3, 0, 3, 1, 2, 1])
        ]
        self.loss.center_region_shrink_ratio = 1.0
        (center_region_mask, radius_map, sin_map,
         cos_map) = self.loss._generate_center_mask_attrib_maps(
             img_size, text_polys)
        target = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.assertTrue(np.allclose(center_region_mask, target))
        self.assertTrue(np.allclose(sin_map, np.zeros(img_size)))
        self.assertTrue(np.allclose(cos_map, target))

    def test_get_targets(self):
        targets = self.loss.get_targets(self.data_samples)
        for target in targets:
            self.assertEqual(len(target), 1)
