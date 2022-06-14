# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch

from mmocr.core.data_structures import TextRecogDataSample
from mmocr.models.textrecog.encoders import NRTREncoder


class TestNRTREncoder(unittest.TestCase):

    def setUp(self):
        self.feat = torch.randn(2, 512, 8, 25)
        gt_text_sample1 = TextRecogDataSample()
        gt_text_sample1.set_metainfo(dict(valid_ratio=0.9))

        gt_text_sample2 = TextRecogDataSample()
        gt_text_sample2.set_metainfo(dict(valid_ratio=1.0))

        self.data_info = [gt_text_sample1, gt_text_sample2]

    def test_encoder(self):
        nrtr_encoder = NRTREncoder()
        nrtr_encoder.init_weights()
        nrtr_encoder.train()
        out_enc = nrtr_encoder(self.feat, self.data_info)
        self.assertEqual(out_enc.shape, torch.Size([2, 200, 512]))
