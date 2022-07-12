# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase, mock

import torch

from mmocr.data import TextRecogDataSample
from mmocr.models.textrecog.dictionary import Dictionary
from mmocr.models.textrecog.postprocessors import BaseTextRecogPostprocessor


class TestBaseTextRecogPostprocessor(TestCase):

    def _create_dummy_dict_file(
        self, dict_file,
        chars=list('0123456789abcdefghijklmnopqrstuvwxyz')):  # NOQA
        with open(dict_file, 'w') as f:
            for char in chars:
                f.write(char + '\n')

    def test_init(self):
        tmp_dir = tempfile.TemporaryDirectory()
        dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
        self._create_dummy_dict_file(dict_file)
        # test diction cfg
        dict_cfg = dict(
            type='Dictionary',
            dict_file=dict_file,
            with_start=True,
            with_end=True,
            same_start_end=False,
            with_padding=True,
            with_unknown=True)
        base_postprocessor = BaseTextRecogPostprocessor(dict_cfg)
        self.assertIsInstance(base_postprocessor.dictionary, Dictionary)
        self.assertListEqual(base_postprocessor.ignore_indexes,
                             [base_postprocessor.dictionary.padding_idx])

        base_postprocessor = BaseTextRecogPostprocessor(
            dict_cfg, ignore_chars=['1', '2', '3'])

        self.assertListEqual(base_postprocessor.ignore_indexes, [1, 2, 3])

        # test ignore_chars
        with self.assertRaisesRegex(TypeError,
                                    'ignore_chars must be list of str'):
            base_postprocessor = BaseTextRecogPostprocessor(
                dict_cfg, ignore_chars=[1, 2, 3])
        with self.assertWarnsRegex(Warning,
                                   'M does not exist in the dictionary'):
            base_postprocessor = BaseTextRecogPostprocessor(
                dict_cfg, ignore_chars=['M'])

        base_postprocessor = BaseTextRecogPostprocessor(
            dict_cfg, ignore_chars=['1', '2', '3'])
        # test dictionary is invalid type
        dict_cfg = ['tmp']
        with self.assertRaisesRegex(
                TypeError, ('The type of dictionary should be `Dictionary`'
                            ' or dict, '
                            f'but got {type(dict_cfg)}')):
            base_postprocessor = BaseTextRecogPostprocessor(dict_cfg)

        tmp_dir.cleanup()

    @mock.patch(f'{__name__}.BaseTextRecogPostprocessor.get_single_prediction')
    def test_call(self, mock_get_single_prediction):

        def mock_func(output, data_sample):
            return [0, 1, 2], [0.8, 0.7, 0.9]

        tmp_dir = tempfile.TemporaryDirectory()
        dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
        self._create_dummy_dict_file(dict_file)
        dict_cfg = dict(
            type='Dictionary',
            dict_file=dict_file,
            with_start=True,
            with_end=True,
            same_start_end=False,
            with_padding=True,
            with_unknown=True)
        mock_get_single_prediction.side_effect = mock_func
        data_samples = [TextRecogDataSample()]
        postprocessor = BaseTextRecogPostprocessor(
            max_seq_len=None, dictionary=dict_cfg)

        # test decode output to index
        dummy_output = torch.Tensor([[[1, 100, 3, 4, 5, 6, 7, 8]]])
        data_samples = postprocessor(dummy_output, data_samples)
        self.assertEqual(data_samples[0].pred_text.item, '012')
        tmp_dir.cleanup()
