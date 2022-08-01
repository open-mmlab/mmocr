# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.textrecog.recognizers import EncoderDecoderRecognizer
from mmocr.registry import MODELS


class TestEncoderDecoderRecognizer(TestCase):

    @MODELS.register_module()
    class DummyModule:

        def __init__(self, value):
            self.value = value

        def __call__(self, x, *args, **kwargs):
            return x + self.value

        def predict(self, x, y, *args, **kwargs):
            if y is None:
                return x
            return x + y

        def loss(self, x, y, *args, **kwargs):
            if y is None:
                return x
            return x * y

    def test_init(self):
        # Decoder is not allowed to be None
        with self.assertRaises(AssertionError):
            EncoderDecoderRecognizer()

        for attr in ['backbone', 'preprocessor', 'encoder']:
            recognizer = EncoderDecoderRecognizer(
                **{
                    attr: dict(type='DummyModule', value=1),
                    'decoder': dict(type='DummyModule', value=2)
                })
            self.assertTrue(hasattr(recognizer, attr))
            self.assertFalse(
                any(
                    hasattr(recognizer, t)
                    for t in ['backbone', 'preprocessor', 'encoder']
                    if t != attr))

    def test_extract_feat(self):
        model = EncoderDecoderRecognizer(
            decoder=dict(type='DummyModule', value=1))
        self.assertEqual(
            model.extract_feat(torch.tensor([1])), torch.Tensor([1]))
        model = EncoderDecoderRecognizer(
            backbone=dict(type='DummyModule', value=1),
            decoder=dict(type='DummyModule', value=1))
        self.assertEqual(
            model.extract_feat(torch.tensor([1])), torch.Tensor([2]))
        model = EncoderDecoderRecognizer(
            preprocessor=dict(type='DummyModule', value=2),
            decoder=dict(type='DummyModule', value=1))
        self.assertEqual(
            model.extract_feat(torch.tensor([1])), torch.Tensor([3]))
        model = EncoderDecoderRecognizer(
            preprocessor=dict(type='DummyModule', value=2),
            backbone=dict(type='DummyModule', value=1),
            decoder=dict(type='DummyModule', value=1))
        self.assertEqual(
            model.extract_feat(torch.tensor([1])), torch.Tensor([4]))

    def test_loss(self):
        model = EncoderDecoderRecognizer(
            decoder=dict(type='DummyModule', value=1))
        self.assertEqual(
            model.loss(torch.tensor([1]), None), torch.Tensor([1]))
        model = EncoderDecoderRecognizer(
            encoder=dict(type='DummyModule', value=2),
            decoder=dict(type='DummyModule', value=1))
        self.assertEqual(
            model.loss(torch.tensor([1]), None), torch.Tensor([3]))
        model = EncoderDecoderRecognizer(
            backbone=dict(type='DummyModule', value=1),
            encoder=dict(type='DummyModule', value=2),
            decoder=dict(type='DummyModule', value=1))
        self.assertEqual(
            model.loss(torch.tensor([1]), None), torch.Tensor([8]))
        model = EncoderDecoderRecognizer(
            backbone=dict(type='DummyModule', value=1),
            decoder=dict(type='DummyModule', value=1))
        self.assertEqual(
            model.loss(torch.tensor([1]), None), torch.Tensor([2]))

    def test_predict(self):
        model = EncoderDecoderRecognizer(
            decoder=dict(type='DummyModule', value=1))
        self.assertEqual(
            model.predict(torch.tensor([1]), None), torch.Tensor([1]))
        model = EncoderDecoderRecognizer(
            encoder=dict(type='DummyModule', value=2),
            decoder=dict(type='DummyModule', value=1))
        self.assertEqual(
            model.predict(torch.tensor([1]), None), torch.Tensor([4]))
        model = EncoderDecoderRecognizer(
            backbone=dict(type='DummyModule', value=1),
            encoder=dict(type='DummyModule', value=2),
            decoder=dict(type='DummyModule', value=1))
        self.assertEqual(
            model.predict(torch.tensor([1]), None), torch.Tensor([6]))
        model = EncoderDecoderRecognizer(
            backbone=dict(type='DummyModule', value=1),
            decoder=dict(type='DummyModule', value=1))
        self.assertEqual(
            model.loss(torch.tensor([1]), None), torch.Tensor([2]))

    def test_forward(self):
        model = EncoderDecoderRecognizer(
            decoder=dict(type='DummyModule', value=1))
        self.assertEqual(
            model._forward(torch.tensor([1]), None), torch.Tensor([2]))
        model = EncoderDecoderRecognizer(
            encoder=dict(type='DummyModule', value=2),
            decoder=dict(type='DummyModule', value=1))
        self.assertEqual(
            model._forward(torch.tensor([1]), None), torch.Tensor([2]))
        model = EncoderDecoderRecognizer(
            backbone=dict(type='DummyModule', value=1),
            encoder=dict(type='DummyModule', value=2),
            decoder=dict(type='DummyModule', value=1))
        self.assertEqual(
            model._forward(torch.tensor([1]), None), torch.Tensor([3]))
        model = EncoderDecoderRecognizer(
            backbone=dict(type='DummyModule', value=1),
            decoder=dict(type='DummyModule', value=1))
        self.assertEqual(
            model._forward(torch.tensor([1]), None), torch.Tensor([3]))
