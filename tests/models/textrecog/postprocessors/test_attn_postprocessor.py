# Copyright (c) OpenMMLab. All rights reserved.

import os.path as osp
import tempfile
from unittest import TestCase

import torch

from mmocr.structures import TextRecogDataSample
from mmocr.models.textrecog.dictionary import Dictionary
from mmocr.models.textrecog.postprocessors.attn_postprocessor import \
    AttentionPostprocessor
from mmocr.testing import create_dummy_dict_file


class TestAttentionPostprocessor(TestCase):

    def test_call(self):
        tmp_dir = tempfile.TemporaryDirectory()
        dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
        create_dummy_dict_file(dict_file)
        dict_gen = Dictionary(
            dict_file=dict_file,
            with_start=True,
            with_end=True,
            same_start_end=True,
            with_padding=True,
            with_unknown=False)
        data_samples = [TextRecogDataSample()]
        postprocessor = AttentionPostprocessor(
            max_seq_len=None, dictionary=dict_gen, ignore_chars=['0'])
        dict_gen.end_idx = 3
        # test decode output to index
        dummy_output = torch.Tensor([[[1, 100, 3, 4, 5, 6, 7, 8],
                                      [100, 2, 3, 4, 5, 6, 7, 8],
                                      [1, 2, 100, 4, 5, 6, 7, 8],
                                      [1, 2, 100, 4, 5, 6, 7, 8],
                                      [100, 2, 3, 4, 5, 6, 7, 8],
                                      [1, 2, 3, 100, 5, 6, 7, 8],
                                      [100, 2, 3, 4, 5, 6, 7, 8],
                                      [1, 2, 3, 100, 5, 6, 7, 8]]])
        data_samples = postprocessor(dummy_output, data_samples)
        self.assertEqual(data_samples[0].pred_text.item, '122')
