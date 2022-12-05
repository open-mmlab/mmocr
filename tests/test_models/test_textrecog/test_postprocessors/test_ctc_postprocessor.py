# Copyright (c) OpenMMLab. All rights reserved.

import os.path as osp
import tempfile
from unittest import TestCase

import torch

from mmocr.models.common.dictionary import Dictionary
from mmocr.models.textrecog.postprocessors.ctc_postprocessor import \
    CTCPostProcessor
from mmocr.structures import TextRecogDataSample
from mmocr.testing import create_dummy_dict_file


class TestCTCPostProcessor(TestCase):

    def test_get_single_prediction(self):

        tmp_dir = tempfile.TemporaryDirectory()
        dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
        create_dummy_dict_file(dict_file)
        dict_gen = Dictionary(
            dict_file=dict_file,
            with_start=False,
            with_end=False,
            with_padding=True,
            with_unknown=False)
        data_samples = [TextRecogDataSample()]
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
        create_dummy_dict_file(dict_file)
        dict_gen = Dictionary(
            dict_file=dict_file,
            with_start=False,
            with_end=False,
            with_padding=True,
            with_unknown=False)
        data_samples = [TextRecogDataSample()]
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
