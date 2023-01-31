# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch

from mmengine.structures import LabelData
from mmocr.structures import TextRecogDataSample


class TestTextRecogDataSample(TestCase):

    def test_init(self):
        meta_info = dict(
            img_size=[256, 256],
            scale_factor=np.array([1.5, 1.5]),
            img_shape=torch.rand(4))

        recog_data_sample = TextRecogDataSample(metainfo=meta_info)
        assert 'img_size' in recog_data_sample

        self.assertListEqual(recog_data_sample.img_size, [256, 256])
        self.assertListEqual(recog_data_sample.get('img_size'), [256, 256])

    def test_setter(self):
        recog_data_sample = TextRecogDataSample()
        # test gt_text
        gt_label_data = dict(item='mmocr')
        gt_text = LabelData(**gt_label_data)
        recog_data_sample.gt_text = gt_text
        assert 'gt_text' in recog_data_sample
        self.assertEqual(recog_data_sample.gt_text.item, gt_text.item)

        # test pred_text
        pred_label_data = dict(item='mmocr')
        pred_text = LabelData(**pred_label_data)
        recog_data_sample.pred_text = pred_text
        assert 'pred_text' in recog_data_sample
        self.assertEqual(recog_data_sample.pred_text.item, pred_text.item)
        # test type error
        with self.assertRaises(AssertionError):
            recog_data_sample.gt_text = torch.rand(2, 4)
        with self.assertRaises(AssertionError):
            recog_data_sample.pred_text = torch.rand(2, 4)

    def test_deleter(self):
        recog_data_sample = TextRecogDataSample()
        # test gt_text
        gt_label_data = dict(item='mmocr')
        gt_text = LabelData(**gt_label_data)
        recog_data_sample.gt_text = gt_text
        assert 'gt_text' in recog_data_sample
        del recog_data_sample.gt_text
        assert 'gt_text' not in recog_data_sample

        recog_data_sample.pred_text = gt_text
        assert 'pred_text' in recog_data_sample
        del recog_data_sample.pred_text
        assert 'pred_text' not in recog_data_sample
