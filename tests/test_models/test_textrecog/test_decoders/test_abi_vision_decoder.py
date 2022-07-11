# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

import torch

from mmocr.models.textrecog.decoders import ABIVisionDecoder


class TestABIVisionDecoder(TestCase):

    def _create_dummy_dict_file(
        self, dict_file,
        chars=list('0123456789abcdefghijklmnopqrstuvwxyz')):  # NOQA
        with open(dict_file, 'w') as f:
            for char in chars:
                f.write(char + '\n')

    def test_forward(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dict_file = osp.join(tmp_dir, 'fake_chars.txt')
            self._create_dummy_dict_file(dict_file)
            # test diction cfg
            dict_cfg = dict(
                type='Dictionary',
                dict_file=dict_file,
                with_start=True,
                with_end=True,
                same_start_end=False,
                with_padding=False,
                with_unknown=False)

            decoder = ABIVisionDecoder(
                dict_cfg, in_channels=32, num_channels=16, max_seq_len=10)

            # training
            out_enc = torch.randn(2, 32, 8, 32)
            result = decoder.forward_train(None, out_enc)
            self.assertIn('feature', result)
            self.assertIn('logits', result)
            self.assertIn('attn_scores', result)
            self.assertEqual(result['feature'].shape, torch.Size([2, 10, 32]))
            self.assertEqual(result['logits'].shape, torch.Size([2, 10, 38]))
            self.assertEqual(result['attn_scores'].shape,
                             torch.Size([2, 10, 8, 32]))

            # testing
            result = decoder.forward_test(None, out_enc)
            self.assertIn('feature', result)
            self.assertIn('logits', result)
            self.assertIn('attn_scores', result)
            self.assertEqual(result['feature'].shape, torch.Size([2, 10, 32]))
            self.assertEqual(result['logits'].shape, torch.Size([2, 10, 38]))
            self.assertEqual(result['attn_scores'].shape,
                             torch.Size([2, 10, 8, 32]))
