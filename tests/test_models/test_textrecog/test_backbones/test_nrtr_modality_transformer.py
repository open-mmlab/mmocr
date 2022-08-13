# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch

from mmocr.models.textrecog.backbones import NRTRModalityTransform


class TestNRTRBackbone(unittest.TestCase):

    def setUp(self):
        self.img = torch.randn(2, 3, 32, 100)

    def test_encoder(self):
        nrtr_backbone = NRTRModalityTransform()
        nrtr_backbone.init_weights()
        nrtr_backbone.train()
        out_enc = nrtr_backbone(self.img)
        self.assertEqual(out_enc.shape, torch.Size([2, 512, 1, 25]))
