# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

import torch

from mmocr.models.textrecog.decoders import ABIFuser


class TestABINetFuser(TestCase):

    def _create_dummy_dict_file(
        self, dict_file,
        chars=list('0123456789abcdefghijklmnopqrstuvwxyz')):  # NOQA
        with open(dict_file, 'w') as f:
            for char in chars:
                f.write(char + '\n')

    def setUp(self):

        self.tmp_dir = tempfile.TemporaryDirectory()
        self.dict_file = osp.join(self.tmp_dir.name, 'fake_chars.txt')
        self._create_dummy_dict_file(self.dict_file)
        self.dict_cfg = dict(
            type='Dictionary',
            dict_file=self.dict_file,
            with_start=True,
            with_end=True,
            same_start_end=False,
            with_padding=False,
            with_unknown=False)

        # both max_seq_len has been set
        with self.assertWarns(Warning):
            ABIFuser(
                self.dict_cfg,
                max_seq_len=10,
                vision_decoder=dict(
                    type='ABIVisionDecoder',
                    in_channels=2,
                    num_channels=2,
                    max_seq_len=5),
                language_decoder=dict(
                    type='ABILanguageDecoder',
                    d_model=2,
                    n_head=2,
                    d_inner=16,
                    n_layers=1,
                    max_seq_len=5))

        # both dictionaries have been set
        with self.assertWarns(Warning):
            ABIFuser(
                self.dict_cfg,
                max_seq_len=10,
                vision_decoder=dict(
                    type='ABIVisionDecoder',
                    in_channels=2,
                    num_channels=2,
                    dictionary=self.dict_cfg),
                language_decoder=dict(
                    type='ABILanguageDecoder',
                    d_model=2,
                    n_head=2,
                    d_inner=16,
                    n_layers=1,
                    dictionary=self.dict_cfg))

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_init(self):
        # No ending idx
        with self.assertRaises(AssertionError):
            dict_cfg = dict(
                type='Dictionary', dict_file=self.dict_file, with_end=False)
            ABIFuser(dict_cfg, None)

    def test_forward_full_model(self):
        # Full model
        model = ABIFuser(
            self.dict_cfg,
            max_seq_len=10,
            vision_decoder=dict(
                type='ABIVisionDecoder', in_channels=2, num_channels=2),
            language_decoder=dict(
                type='ABILanguageDecoder',
                d_model=2,
                n_head=2,
                d_inner=16,
                n_layers=1,
            ),
            d_model=2)
        model.train()
        result = model(None, torch.randn(1, 2, 8, 32))
        self.assertIsInstance(result, dict)
        self.assertIn('out_vis', result)
        self.assertIn('out_langs', result)
        self.assertIsInstance(result['out_langs'], list)
        self.assertIn('out_fusers', result)
        self.assertIsInstance(result['out_fusers'], list)

        model.eval()
        result = model(None, torch.randn(1, 2, 8, 32))
        self.assertIsInstance(result, torch.Tensor)

    def test_forward_vision_model(self):
        # Full model
        model = ABIFuser(
            self.dict_cfg,
            vision_decoder=dict(
                type='ABIVisionDecoder', in_channels=2, num_channels=2))
        model.train()
        result = model(None, torch.randn(1, 2, 8, 32))
        self.assertIsInstance(result, dict)
        self.assertIn('out_vis', result)
        self.assertIn('out_langs', result)
        self.assertIsInstance(result['out_langs'], list)
        self.assertEqual(len(result['out_langs']), 0)
        self.assertIn('out_fusers', result)
        self.assertIsInstance(result['out_fusers'], list)
        self.assertEqual(len(result['out_fusers']), 0)

        model.eval()
        result = model(None, torch.randn(1, 2, 8, 32))
        self.assertIsInstance(result, torch.Tensor)
