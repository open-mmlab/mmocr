# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch

from mmocr.models.textrecog.encoders import ASTEREncoder


class TestASTEREncoder(unittest.TestCase):

    def test_encoder(self):
        encoder = ASTEREncoder(10)
        feat = torch.randn(2, 10, 1, 25)
        out = encoder(feat)
        self.assertEqual(out.shape, torch.Size([2, 25, 10]))
