# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine import InstanceData

from mmocr.models.textdet.module_losses import DRRGModuleLoss
from mmocr.structures import TextDetDataSample


class TestDRRGModuleLoss(TestCase):

    def setUp(self) -> None:
        preds_maps = torch.rand(1, 6, 64, 64)
        gcn_pred = torch.rand(1, 2)
        gt_labels = torch.zeros((1), dtype=torch.long)
        self.preds = (preds_maps, gcn_pred, gt_labels)
        self.data_samples = [
            TextDetDataSample(
                metainfo=dict(img_shape=(64, 64)),
                gt_instances=InstanceData(
                    polygons=[
                        np.array([4, 2, 30, 2, 30, 10, 4, 10]),
                        np.array([36, 12, 8, 12, 8, 22, 36, 22]),
                        np.array([48, 20, 52, 20, 52, 50, 48, 50]),
                        np.array([44, 50, 38, 50, 38, 20, 44, 20])
                    ],
                    ignored=torch.BoolTensor([False, False, False, False])))
        ]

    def test_forward(self):
        loss = DRRGModuleLoss()
        loss_output = loss(self.preds, self.data_samples)
        self.assertIsInstance(loss_output, dict)
        self.assertIn('loss_text', loss_output)
        self.assertIn('loss_center', loss_output)
        self.assertIn('loss_height', loss_output)
        self.assertIn('loss_sin', loss_output)
        self.assertIn('loss_cos', loss_output)
        self.assertIn('loss_gcn', loss_output)

    def test_get_targets(self):
        # test get_targets
        loss = DRRGModuleLoss(
            min_width=2.,
            max_width=4.,
            min_rand_half_height=3.,
            max_rand_half_height=5.)
        targets = loss.get_targets(self.data_samples)
        for target in targets[:-1]:
            self.assertEqual(len(target), 1)
        self.assertEqual(targets[-1][0].shape[-1], 8)

        # test generate_targets with blank polygon masks
        blank_data_samples = [
            TextDetDataSample(
                metainfo=dict(img_shape=(20, 20)),
                gt_instances=InstanceData(
                    polygons=[], ignored=torch.BoolTensor([])))
        ]
        targets = loss.get_targets(blank_data_samples)
        self.assertGreater(targets[-1][0][0, 0], 8)

        # test get_targets with the number of proposed text components exceeds
        # num_max_comps
        loss = DRRGModuleLoss(
            min_width=2.,
            max_width=4.,
            min_rand_half_height=3.,
            max_rand_half_height=5.,
            num_max_comps=6)
        targets = loss.get_targets(self.data_samples)
        self.assertEqual(targets[-1][0].ndim, 2)
        self.assertEqual(targets[-1][0].shape[0], 6)

        # test generate_targets with one proposed text component
        data_samples = [
            TextDetDataSample(
                metainfo=dict(img_shape=(20, 30)),
                gt_instances=InstanceData(
                    polygons=[np.array([13, 6, 17, 6, 17, 14, 13, 14])],
                    ignored=torch.BoolTensor([False])))
        ]
        loss = DRRGModuleLoss(
            min_width=4.,
            max_width=8.,
            min_rand_half_height=3.,
            max_rand_half_height=5.)
        targets = loss.get_targets(data_samples)
        self.assertGreater(targets[-1][0][0, 0], 8)

        # test generate_targets with shrunk margin in
        # generate_rand_comp_attribs
        loss = DRRGModuleLoss(
            min_width=2.,
            max_width=30.,
            min_rand_half_height=3.,
            max_rand_half_height=30.)
        targets = loss.get_targets(data_samples)
        self.assertGreater(targets[-1][0][0, 0], 8)
