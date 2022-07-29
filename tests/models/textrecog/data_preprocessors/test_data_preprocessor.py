# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.structures import TextRecogDataSample
from mmocr.models.textrecog.data_preprocessors import TextRecogDataPreprocessor
from mmocr.registry import MODELS


@MODELS.register_module()
class Augment(torch.nn.Module):

    def forward(self, batch_inputs, batch_data_samples):
        return batch_inputs, batch_data_samples


class TestTextRecogDataPreprocessor(TestCase):

    def test_init(self):
        # test mean is None
        processor = TextRecogDataPreprocessor()
        self.assertTrue(not hasattr(processor, 'mean'))
        self.assertTrue(processor._enable_normalize is False)

        # test mean is not None
        processor = TextRecogDataPreprocessor(mean=[0, 0, 0], std=[1, 1, 1])
        self.assertTrue(hasattr(processor, 'mean'))
        self.assertTrue(hasattr(processor, 'std'))
        self.assertTrue(processor._enable_normalize)

        # please specify both mean and std
        with self.assertRaises(AssertionError):
            TextRecogDataPreprocessor(mean=[0, 0, 0])

        # bgr2rgb and rgb2bgr cannot be set to True at the same time
        with self.assertRaises(AssertionError):
            TextRecogDataPreprocessor(bgr_to_rgb=True, rgb_to_bgr=True)

        aug_cfg = [dict(type='Augment')]
        processor = TextRecogDataPreprocessor()
        self.assertIsNone(processor.batch_augments)
        processor = TextRecogDataPreprocessor(batch_augments=aug_cfg)
        self.assertIsInstance(processor.batch_augments, torch.nn.ModuleList)
        self.assertIsInstance(processor.batch_augments[0], Augment)

    def test_forward(self):
        processor = TextRecogDataPreprocessor(mean=[0, 0, 0], std=[1, 1, 1])

        data = [{
            'inputs':
            torch.randint(0, 256, (3, 11, 10)),
            'data_sample':
            TextRecogDataSample(
                metainfo=dict(img_shape=(11, 10), valid_ratio=1.0))
        }]
        inputs, data_samples = processor(data)
        print(inputs.dtype)
        self.assertEqual(inputs.shape, (1, 3, 11, 10))
        self.assertEqual(len(data_samples), 1)

        # test channel_conversion
        processor = TextRecogDataPreprocessor(
            mean=[0., 0., 0.], std=[1., 1., 1.], bgr_to_rgb=True)
        inputs, data_samples = processor(data)
        self.assertEqual(inputs.shape, (1, 3, 11, 10))
        self.assertEqual(len(data_samples), 1)

        # test padding
        data = [{
            'inputs': torch.randint(0, 256, (3, 10, 11))
        }, {
            'inputs': torch.randint(0, 256, (3, 9, 14))
        }]
        processor = TextRecogDataPreprocessor(
            mean=[0., 0., 0.], std=[1., 1., 1.], bgr_to_rgb=True)
        inputs, data_samples = processor(data)
        self.assertEqual(inputs.shape, (2, 3, 10, 14))
        self.assertIsNone(data_samples)

        # test pad_size_divisor
        data = [{
            'inputs':
            torch.randint(0, 256, (3, 10, 11)),
            'data_sample':
            TextRecogDataSample(
                metainfo=dict(img_shape=(10, 11), valid_ratio=1.0))
        }, {
            'inputs':
            torch.randint(0, 256, (3, 9, 24)),
            'data_sample':
            TextRecogDataSample(
                metainfo=dict(img_shape=(9, 24), valid_ratio=1.0))
        }]
        aug_cfg = [dict(type='Augment')]
        processor = TextRecogDataPreprocessor(
            mean=[0., 0., 0.],
            std=[1., 1., 1.],
            pad_size_divisor=5,
            batch_augments=aug_cfg)
        inputs, data_samples = processor(data, training=True)
        self.assertEqual(inputs.shape, (2, 3, 10, 25))
        self.assertEqual(len(data_samples), 2)
        for data_sample, expected_shape, expected_ratio in zip(
                data_samples, [(10, 25), (10, 25)], [11 / 25., 24 / 25.]):
            self.assertEqual(data_sample.batch_input_shape, expected_shape)
            self.assertEqual(data_sample.valid_ratio, expected_ratio)
