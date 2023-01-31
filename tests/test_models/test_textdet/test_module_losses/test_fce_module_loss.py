# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine.structures import InstanceData

from mmocr.models.textdet.module_losses import FCEModuleLoss
from mmocr.structures import TextDetDataSample


class TestFCEModuleLoss(TestCase):

    def setUp(self) -> None:
        self.fce_loss = FCEModuleLoss(fourier_degree=5, num_sample=400)
        self.data_samples = [
            TextDetDataSample(
                metainfo=dict(img_shape=(320, 320)),
                gt_instances=InstanceData(
                    polygons=np.array([
                        [0, 0, 10, 0, 10, 10, 0, 10],
                        [20, 0, 30, 0, 30, 10, 20, 10],
                        [0, 0, 15, 0, 15, 10, 0, 10],
                    ],
                                      dtype=np.float32),
                    ignored=torch.BoolTensor([False, False, True])))
        ]
        self.preds = [
            dict(
                cls_res=torch.rand(1, 4, 40, 40),
                reg_res=torch.rand(1, 22, 40, 40)),
            dict(
                cls_res=torch.rand(1, 4, 20, 20),
                reg_res=torch.rand(1, 22, 20, 20)),
            dict(
                cls_res=torch.rand(1, 4, 10, 10),
                reg_res=torch.rand(1, 22, 10, 10))
        ]

    def test_forward(self):
        losses = self.fce_loss(self.preds, self.data_samples)
        assert 'loss_text' in losses
        assert 'loss_center' in losses
        assert 'loss_reg_x' in losses
        assert 'loss_reg_y' in losses
