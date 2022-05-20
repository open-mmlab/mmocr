# Copyright (c) OpenMMLab. All rights reserved.

import os.path as osp
import tempfile
from unittest import TestCase

import torch
from mmengine.data import LabelData

from mmocr.core.data_structures import TextRecogDataSample
from mmocr.models.textrecog.dictionary import Dictionary
from mmocr.models.textrecog.postprocessor.ctc_postprocessor import \
    CTCPostProcessor


class TestCTCPostProcessor(TestCase):

    def _create_dummy_dict_file(
        self, dict_file,
        chars=list('0123456789abcdefghijklmnopqrstuvwxyz')):  # NOQA
        with open(dict_file, 'w') as f:
            for char in chars:
                f.write(char + '\n')

    def test_get_single_prediction(self):

        tmp_dir = tempfile.TemporaryDirectory()
        dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
        self._create_dummy_dict_file(dict_file)
        dict_gen = Dictionary(
            dict_file=dict_file,
            with_start=False,
            with_end=False,
            with_padding=True,
            with_unknown=False)
        pred_text = LabelData(valid_ratio=1.0)
        data_samples = [TextRecogDataSample(pred_text=pred_text)]
        postprocessor = CTCPostProcessor(max_seq_len=None, dictionary=dict_gen)

        # test decode output to index
        dummy_output = torch.Tensor([[[1, 100, 3, 4, 5, 6, 7, 8],
                                      [100, 2, 3, 4, 5, 6, 7, 8],
                                      [1, 2, 100, 4, 5, 6, 7, 8],
                                      [1, 2, 100, 4, 5, 6, 7, 8],
                                      [100, 2, 3, 4, 5, 6, 7, 8],
                                      [1, 2, 3, 100, 5, 6, 7, 8],
                                      [100, 2, 3, 4, 5, 6, 7, 8],
                                      [1, 2, 3, 100, 5, 6, 7, 8]]])
        index, score = postprocessor.get_single_prediction(
            dummy_output[0], data_samples[0])
        self.assertListEqual(index, [1, 0, 2, 0, 3, 0, 3])
        self.assertListEqual(score,
                             [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        postprocessor = CTCPostProcessor(
            max_seq_len=None, dictionary=dict_gen, ignore_chars=['0'])
        index, score = postprocessor.get_single_prediction(
            dummy_output[0], data_samples[0])
        self.assertListEqual(index, [1, 2, 3, 3])
        self.assertListEqual(score, [100.0, 100.0, 100.0, 100.0])
        tmp_dir.cleanup()

    def test_call(self):
        tmp_dir = tempfile.TemporaryDirectory()
        dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
        self._create_dummy_dict_file(dict_file)
        dict_gen = Dictionary(
            dict_file=dict_file,
            with_start=False,
            with_end=False,
            with_padding=True,
            with_unknown=False)
        pred_text = LabelData(valid_ratio=1.0)
        data_samples = [TextRecogDataSample(pred_text=pred_text)]
        postprocessor = CTCPostProcessor(max_seq_len=None, dictionary=dict_gen)

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
        self.assertEqual(data_samples[0].pred_text.item, '1020303')
