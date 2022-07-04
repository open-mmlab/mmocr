# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase, mock

from mmocr.models.textdet import BaseTextDetHead
from mmocr.registry import MODELS


@MODELS.register_module()
class FakeModule:

    def __init__(self) -> None:
        pass

    def get_targets(self, datasamples):
        return None

    def __call__(self, *args):
        return None


class TestBaseTextDetHead(TestCase):

    def test_init(self):
        cfg = dict(type='FakeModule')

        with self.assertRaises(AssertionError):
            BaseTextDetHead([], cfg)
        with self.assertRaises(AssertionError):
            BaseTextDetHead(cfg, [])

        decoder = BaseTextDetHead(cfg, cfg)
        self.assertIsInstance(decoder.loss_module, FakeModule)
        self.assertIsInstance(decoder.postprocessor, FakeModule)

    @mock.patch(f'{__name__}.BaseTextDetHead.forward')
    def test_forward(self, mock_forward):

        def mock_forward(feat, out_enc, datasamples):

            return True

        mock_forward.side_effect = mock_forward
        cfg = dict(type='FakeModule')
        decoder = BaseTextDetHead(cfg, cfg)
        # test loss
        loss = decoder.loss(None, None)
        self.assertIsNone(loss)

        # test predict
        predict = decoder.predict(None, None)
        self.assertIsNone(predict)

        # test forward
        tensor = decoder(None, None)
        self.assertTrue(tensor)

        loss, predict = decoder.loss_and_predict(None, None)
        self.assertIsNone(loss)
        self.assertIsNone(predict)
