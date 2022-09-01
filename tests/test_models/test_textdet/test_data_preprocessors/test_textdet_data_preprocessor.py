# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.textdet.data_preprocessors import TextDetDataPreprocessor
from mmocr.registry import MODELS
from mmocr.structures import TextDetDataSample


@MODELS.register_module()
class TDAugment(torch.nn.Module):

    def forward(self, inputs, data_samples):
        return inputs, data_samples


class TestTextDetDataPreprocessor(TestCase):

    def test_init(self):
        # test mean is None
        processor = TextDetDataPreprocessor()
        self.assertTrue(not hasattr(processor, 'mean'))
        self.assertTrue(processor._enable_normalize is False)

        # test mean is not None
        processor = TextDetDataPreprocessor(mean=[0, 0, 0], std=[1, 1, 1])
        self.assertTrue(hasattr(processor, 'mean'))
        self.assertTrue(hasattr(processor, 'std'))
        self.assertTrue(processor._enable_normalize)

        # please specify both mean and std
        with self.assertRaises(AssertionError):
            TextDetDataPreprocessor(mean=[0, 0, 0])

        # bgr2rgb and rgb2bgr cannot be set to True at the same time
        with self.assertRaises(AssertionError):
            TextDetDataPreprocessor(bgr_to_rgb=True, rgb_to_bgr=True)

        aug_cfg = [dict(type='TDAugment')]
        processor = TextDetDataPreprocessor()
        self.assertIsNone(processor.batch_augments)
        processor = TextDetDataPreprocessor(batch_augments=aug_cfg)
        self.assertIsInstance(processor.batch_augments, torch.nn.ModuleList)
        self.assertIsInstance(processor.batch_augments[0], TDAugment)

    def test_forward(self):
        processor = TextDetDataPreprocessor(mean=[0, 0, 0], std=[1, 1, 1])

        data = {
            'inputs': [
                torch.randint(0, 256, (3, 11, 10)),
            ],
            'data_samples': [
                TextDetDataSample(
                    metainfo=dict(img_shape=(11, 10), valid_ratio=1.0)),
            ]
        }
        out = processor(data)
        inputs, data_samples = out['inputs'], out['data_samples']
        self.assertEqual(inputs.shape, (1, 3, 11, 10))
        self.assertEqual(len(data_samples), 1)

        # test channel_conversion
        processor = TextDetDataPreprocessor(
            mean=[0., 0., 0.], std=[1., 1., 1.], bgr_to_rgb=True)
        out = processor(data)
        inputs, data_samples = out['inputs'], out['data_samples']
        self.assertEqual(inputs.shape, (1, 3, 11, 10))
        self.assertEqual(len(data_samples), 1)

        # test padding
        data = {
            'inputs': [
                torch.randint(0, 256, (3, 10, 11)),
                torch.randint(0, 256, (3, 9, 14))
            ]
        }
        processor = TextDetDataPreprocessor(
            mean=[0., 0., 0.], std=[1., 1., 1.], bgr_to_rgb=True)
        out = processor(data)
        inputs, data_samples = out['inputs'], out['data_samples']
        self.assertEqual(inputs.shape, (2, 3, 10, 14))
        self.assertIsNone(data_samples)

        # test pad_size_divisor
        data = {
            'inputs': [
                torch.randint(0, 256, (3, 10, 11)),
                torch.randint(0, 256, (3, 9, 24))
            ],
            'data_samples': [
                TextDetDataSample(
                    metainfo=dict(img_shape=(10, 11), valid_ratio=1.0)),
                TextDetDataSample(
                    metainfo=dict(img_shape=(9, 24), valid_ratio=1.0))
            ]
        }
        aug_cfg = [dict(type='TDAugment')]
        processor = TextDetDataPreprocessor(
            mean=[0., 0., 0.],
            std=[1., 1., 1.],
            pad_size_divisor=5,
            batch_augments=aug_cfg)
        out = processor(data)
        inputs, data_samples = out['inputs'], out['data_samples']
        self.assertEqual(inputs.shape, (2, 3, 10, 25))
        self.assertEqual(len(data_samples), 2)
        for data_sample, expected_shape in zip(data_samples, [(10, 25),
                                                              (10, 25)]):
            self.assertEqual(data_sample.batch_input_shape, expected_shape)
