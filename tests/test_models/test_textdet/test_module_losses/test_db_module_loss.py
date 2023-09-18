# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine.structures import InstanceData

from mmocr.models.textdet.module_losses import DBModuleLoss
from mmocr.structures import TextDetDataSample


class TestDBModuleLoss(TestCase):

    def setUp(self) -> None:
        self.db_loss = DBModuleLoss(thr_min=0.3, thr_max=0.7)
        self.data_samples = [
            TextDetDataSample(
                metainfo=dict(img_shape=(40, 40), batch_input_shape=(40, 40)),
                gt_instances=InstanceData(
                    polygons=np.array([
                        [0, 0, 10, 0, 10, 10, 0, 10],
                        [20, 0, 30, 0, 30, 10, 20, 10],
                        [0, 0, 15, 0, 15, 10, 0, 10],
                    ],
                                      dtype=np.float32),
                    ignored=torch.BoolTensor([False, False, True])))
        ]
        pred_size = (1, 40, 40)
        self.preds = (torch.rand(pred_size), torch.rand(pred_size),
                      torch.rand(pred_size))

    def test_is_poly_invalid(self):
        # area < 1
        poly = np.array([0, 0, 0.5, 0, 0.5, 0.5, 0, 0.5], dtype=np.float32)
        self.assertTrue(self.db_loss._is_poly_invalid(poly))

        # Sidelength < min_sidelength
        # area < 1
        poly = np.array([0.5, 0.5, 2.5, 2.5, 2, 3, 0, 1], dtype=np.float32)
        self.assertTrue(self.db_loss._is_poly_invalid(poly))

        # A good enough polygon
        poly = np.array([0, 0, 10, 0, 10, 10, 0, 10], dtype=np.float32)
        self.assertFalse(self.db_loss._is_poly_invalid(poly))

    def test_draw_border_map(self):
        img_size = (40, 40)
        thr_map = np.zeros(img_size, dtype=np.float32)
        thr_mask = np.zeros(img_size, dtype=np.float32)
        polygon = np.array([20, 21, -14, 20, -11, 30, -22, 26],
                           dtype=np.float32)
        self.db_loss._draw_border_map(polygon, thr_map, thr_mask)

    def test_generate_thr_map(self):
        data_sample = self.data_samples[0]
        text_polys = data_sample.gt_instances.polygons[:2]
        thr_map, _ = self.db_loss._generate_thr_map(data_sample.img_shape,
                                                    text_polys)
        assert np.all((thr_map >= 0.29) * (thr_map <= 0.71))

    def test_forward(self):
        losses = self.db_loss(self.preds, self.data_samples)
        assert 'loss_prob' in losses
        assert 'loss_thr' in losses
        assert 'loss_db' in losses
