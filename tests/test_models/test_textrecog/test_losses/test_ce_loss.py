# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.data import LabelData

from mmocr.data import TextRecogDataSample
from mmocr.models.textrecog.losses import CELoss


class TestCELoss(TestCase):

    def setUp(self) -> None:

        data_sample1 = TextRecogDataSample()
        data_sample1.gt_text = LabelData(item='hello')
        data_sample2 = TextRecogDataSample()
        data_sample2.gt_text = LabelData(item='01abyz')
        data_sample3 = TextRecogDataSample()
        data_sample3.gt_text = LabelData(item='123456789')
        self.gt = [data_sample1, data_sample2, data_sample3]

    def test_init(self):
        dict_cfg = dict(
            type='Dictionary',
            dict_file='dicts/lower_english_digits.txt',
            with_start=True,
            with_end=True,
            same_start_end=True,
            with_padding=True,
            with_unknown=False)

        with self.assertRaises(AssertionError):
            CELoss(dict_cfg, reduction=1)
        with self.assertRaises(AssertionError):
            CELoss(dict_cfg, reduction='avg')
        with self.assertRaises(AssertionError):
            CELoss(dict_cfg, flatten=1)
        with self.assertRaises(AssertionError):
            CELoss(dict_cfg, ignore_first_char=1)
        with self.assertRaises(AssertionError):
            CELoss(dict_cfg, ignore_char=['ignore'])
        ce_loss = CELoss(dict_cfg)
        self.assertEqual(ce_loss.ignore_index, 37)
        ce_loss = CELoss(dict_cfg, ignore_char=-1)
        self.assertEqual(ce_loss.ignore_index, -1)
        with self.assertRaises(ValueError):
            ce_loss = CELoss(dict_cfg, ignore_char='ignore')
        ce_loss = CELoss(dict_cfg, ignore_char='1')
        self.assertEqual(ce_loss.ignore_index, 1)

    def test_ce_loss(self):
        dict_cfg = dict(
            type='Dictionary',
            dict_file='dicts/lower_english_digits.txt',
            with_start=True,
            with_end=True,
            same_start_end=True,
            with_padding=True,
            with_unknown=False)
        max_seq_len = 40
        ce_loss = CELoss(dict_cfg)
        ce_loss.get_targets(self.gt)
        outputs = torch.rand(3, max_seq_len, ce_loss.dictionary.num_classes)
        losses = ce_loss(outputs, self.gt)
        self.assertIsInstance(losses, dict)
        self.assertIn('loss_ce', losses)
        self.assertEqual(losses['loss_ce'].size(1), max_seq_len)

        # test ignore_first_char
        ce_loss = CELoss(dict_cfg, ignore_first_char=True)
        ignore_first_char_losses = ce_loss(outputs, self.gt)
        self.assertEqual(ignore_first_char_losses['loss_ce'].shape,
                         torch.Size([3, max_seq_len - 1]))

        # test flatten
        ce_loss = CELoss(dict_cfg, flatten=True)
        flatten_losses = ce_loss(outputs, self.gt)
        self.assertEqual(flatten_losses['loss_ce'].shape,
                         torch.Size([3 * max_seq_len]))

        self.assertTrue(
            torch.isclose(
                losses['loss_ce'].view(-1),
                flatten_losses['loss_ce'],
                atol=1e-6,
                rtol=0).all())
