# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase, mock

from mmocr.models.textrecog.decoders import BaseDecoder
from mmocr.registry import MODELS


@MODELS.register_module()
class Tmp:
    pass


class TestBaseDecoder(TestCase):

    def test_init(self):
        cfg = dict(type='Tmp')
        with self.assertRaises(AssertionError):
            BaseDecoder([], cfg)
        with self.assertRaises(AssertionError):
            BaseDecoder(cfg, [])
        decoder = BaseDecoder()
        self.assertIsNone(decoder.loss)
        self.assertIsNone(decoder.postprocessor)

        decoder = BaseDecoder(cfg, cfg)
        self.assertIsInstance(decoder.loss, Tmp)
        self.assertIsInstance(decoder.postprocessor, Tmp)

    def test_forward_train(self):
        decoder = BaseDecoder()
        with self.assertRaises(NotImplementedError):
            decoder.forward_train(None, None, None)

    def test_forward_test(self):
        decoder = BaseDecoder()
        with self.assertRaises(NotImplementedError):
            decoder.forward_test(None, None, None)

    @mock.patch(f'{__name__}.BaseDecoder.forward_test')
    @mock.patch(f'{__name__}.BaseDecoder.forward_train')
    def test_forward(self, mock_forward_train, mock_forward_test):

        def mock_func_train(feat, out_enc, datasamples):
            return True

        def mock_func_test(feat, out_enc, datasamples):
            return False

        mock_forward_train.side_effect = mock_func_train
        mock_forward_test.side_effect = mock_func_test
        cfg = dict(type='Tmp')
        decoder = BaseDecoder(cfg, cfg)

        self.assertTrue(decoder(None, None, None, True))
        self.assertFalse(decoder(None, None, None, False))
