# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase, mock

from mmocr.models.textrecog.decoders import BaseDecoder
from mmocr.models.textrecog.dictionary.dictionary import Dictionary
from mmocr.registry import MODELS, TASK_UTILS
from mmocr.testing import create_dummy_dict_file


@MODELS.register_module()
class Tmp:

    def __init__(self, max_seq_len, dictionary) -> None:
        pass

    def get_targets(self, datasamples):
        return None

    def __call__(self, *args):
        return None


class TestBaseDecoder(TestCase):

    def test_init(self):
        cfg = dict(type='Tmp')
        tmp_dir = tempfile.TemporaryDirectory()
        dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
        create_dummy_dict_file(dict_file)
        # test diction cfg
        dict_cfg = dict(
            type='Dictionary',
            dict_file=dict_file,
            with_start=True,
            with_end=True,
            same_start_end=False,
            with_padding=True,
            with_unknown=True)
        with self.assertRaises(AssertionError):
            BaseDecoder(dict_cfg, [], cfg)
        with self.assertRaises(AssertionError):
            BaseDecoder(dict_cfg, cfg, [])
        with self.assertRaises(TypeError):
            BaseDecoder([], cfg, cfg)
        decoder = BaseDecoder(dictionary=dict_cfg)
        self.assertIsNone(decoder.loss_module)
        self.assertIsNone(decoder.postprocessor)
        self.assertIsInstance(decoder.dictionary, Dictionary)
        decoder = BaseDecoder(dict_cfg, cfg, cfg)
        self.assertIsInstance(decoder.loss_module, Tmp)
        self.assertIsInstance(decoder.postprocessor, Tmp)
        dictionary = TASK_UTILS.build(dict_cfg)
        decoder = BaseDecoder(dictionary, cfg, cfg)
        self.assertIsInstance(decoder.dictionary, Dictionary)
        tmp_dir.cleanup()

    def test_forward_train(self):
        tmp_dir = tempfile.TemporaryDirectory()
        dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
        create_dummy_dict_file(dict_file)
        # test diction cfg
        dict_cfg = dict(
            type='Dictionary',
            dict_file=dict_file,
            with_start=True,
            with_end=True,
            same_start_end=False,
            with_padding=True,
            with_unknown=True)
        decoder = BaseDecoder(dictionary=dict_cfg)
        with self.assertRaises(NotImplementedError):
            decoder.forward_train(None, None, None)
        tmp_dir.cleanup()

    def test_forward_test(self):
        tmp_dir = tempfile.TemporaryDirectory()
        dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
        create_dummy_dict_file(dict_file)
        dict_cfg = dict(
            type='Dictionary',
            dict_file=dict_file,
            with_start=True,
            with_end=True,
            same_start_end=False,
            with_padding=True,
            with_unknown=True)
        decoder = BaseDecoder(dictionary=dict_cfg)
        with self.assertRaises(NotImplementedError):
            decoder.forward_test(None, None, None)
        tmp_dir.cleanup()

    @mock.patch(f'{__name__}.BaseDecoder.forward_test')
    @mock.patch(f'{__name__}.BaseDecoder.forward_train')
    def test_forward(self, mock_forward_train, mock_forward_test):

        def mock_func_train(feat, out_enc, datasamples):

            return True

        def mock_func_test(feat, out_enc, datasamples):

            return False

        tmp_dir = tempfile.TemporaryDirectory()
        dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
        create_dummy_dict_file(dict_file)
        dict_cfg = dict(
            type='Dictionary',
            dict_file=dict_file,
            with_start=True,
            with_end=True,
            same_start_end=False,
            with_padding=True,
            with_unknown=True)
        mock_forward_train.side_effect = mock_func_train
        mock_forward_test.side_effect = mock_func_test
        cfg = dict(type='Tmp')
        decoder = BaseDecoder(dict_cfg, cfg, cfg)
        # test loss
        loss = decoder.loss(None, None, None)
        self.assertIsNone(loss)

        # test predict
        predict = decoder.predict(None, None, None)
        self.assertIsNone(predict)

        # test forward
        tensor = decoder(None, None, None)
        self.assertTrue(tensor)
        decoder.eval()
        tensor = decoder(None, None, None)
        self.assertFalse(tensor)
        tmp_dir.cleanup()
