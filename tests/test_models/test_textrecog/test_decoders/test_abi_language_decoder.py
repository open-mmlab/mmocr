# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

import torch

from mmocr.models.textrecog.decoders import ABILanguageDecoder
from mmocr.testing import create_dummy_dict_file


class TestABILanguageDecoder(TestCase):

    def test_init(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dict_file = osp.join(tmp_dir, 'fake_chars.txt')
            create_dummy_dict_file(dict_file)
            dict_cfg = dict(
                type='Dictionary', dict_file=dict_file, with_end=False)
            # No padding token
            with self.assertRaises(AssertionError):
                ABILanguageDecoder(dict_cfg)

    def test_forward(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dict_file = osp.join(tmp_dir, 'fake_chars.txt')
            create_dummy_dict_file(dict_file)
            # test diction cfg
            dict_cfg = dict(
                type='Dictionary',
                dict_file=dict_file,
                with_start=True,
                with_end=True,
                same_start_end=False,
                with_padding=True,
                with_unknown=False)
            decoder = ABILanguageDecoder(
                dict_cfg, d_model=16, d_inner=16, max_seq_len=10)
            logits = torch.randn(2, 10, 39)
            result = decoder.forward_train(None, logits, None)
            self.assertIn('feature', result)
            self.assertIn('logits', result)
            self.assertEqual(result['feature'].shape, torch.Size([2, 10, 16]))
            self.assertEqual(result['logits'].shape, torch.Size([2, 10, 39]))

            decoder = ABILanguageDecoder(
                dict_cfg,
                d_model=16,
                d_inner=16,
                max_seq_len=10,
                with_self_attn=True,
                detach_tokens=False)
            logits = torch.randn(2, 10, 39)
            result = decoder.forward_test(None, logits, None)
            self.assertIn('feature', result)
            self.assertIn('logits', result)
            self.assertEqual(result['feature'].shape, torch.Size([2, 10, 16]))
            self.assertEqual(result['logits'].shape, torch.Size([2, 10, 39]))
