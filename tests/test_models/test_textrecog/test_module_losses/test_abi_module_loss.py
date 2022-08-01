# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine.data import LabelData

from mmocr.models.textrecog.module_losses import ABIModuleLoss
from mmocr.structures import TextRecogDataSample


class TestABIModuleLoss(TestCase):

    def setUp(self) -> None:

        data_sample1 = TextRecogDataSample()
        data_sample1.gt_text = LabelData(item='hello')
        data_sample2 = TextRecogDataSample()
        data_sample2.gt_text = LabelData(item='123')
        self.gt = [data_sample1, data_sample2]

    def _equal(self, a, b):
        if isinstance(a, (torch.Tensor, np.ndarray)):
            return (a == b).all()
        else:
            return a == b

    def test_forward(self):
        dict_cfg = dict(
            type='Dictionary',
            dict_file='dicts/lower_english_digits.txt',
            with_start=True,
            with_end=True,
            same_start_end=True,
            with_padding=True,
            with_unknown=False)
        abi_loss = ABIModuleLoss(dict_cfg, max_seq_len=10)
        abi_loss.get_targets(self.gt)
        outputs = dict(
            out_vis=dict(logits=torch.randn(2, 10, 38)),
            out_langs=[
                dict(logits=torch.randn(2, 10, 38)),
                dict(logits=torch.randn(2, 10, 38))
            ],
            out_fusers=[
                dict(logits=torch.randn(2, 10, 38)),
                dict(logits=torch.randn(2, 10, 38))
            ])
        losses = abi_loss(outputs, self.gt)
        self.assertIsInstance(losses, dict)
        self.assertIn('loss_visual', losses)
        self.assertIn('loss_lang', losses)
        self.assertIn('loss_fusion', losses)
        print(losses['loss_lang'])
        print(losses['loss_fusion'])

        outputs.pop('out_vis')
        abi_loss(outputs, self.gt)
        out_langs = outputs.pop('out_langs')
        abi_loss(outputs, self.gt)
        outputs.pop('out_fusers')
        with self.assertRaises(AssertionError):
            abi_loss(outputs, self.gt)
        outputs['out_langs'] = out_langs
        abi_loss(outputs, self.gt)
