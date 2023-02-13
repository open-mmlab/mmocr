# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
import torch.nn as nn
from mmengine.structures import LabelData

from mmocr.models.textrecog.recognizers import EncoderDecoderRecognizerTTAModel
from mmocr.structures import TextRecogDataSample


class DummyModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def test_step(self, x):
        return self.forward(x)


class TestEncoderDecoderRecognizerTTAModel(TestCase):

    def test_merge_preds(self):

        data_sample1 = TextRecogDataSample(
            pred_text=LabelData(
                score=torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]), text='abcde'))
        data_sample2 = TextRecogDataSample(
            pred_text=LabelData(
                score=torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6]), text='bcdef'))
        data_sample3 = TextRecogDataSample(
            pred_text=LabelData(
                score=torch.tensor([0.3, 0.4, 0.5, 0.6, 0.7]), text='cdefg'))
        aug_data_samples = [data_sample1, data_sample2, data_sample3]
        batch_aug_data_samples = [aug_data_samples] * 3
        model = EncoderDecoderRecognizerTTAModel(module=DummyModel())
        preds = model.merge_preds(batch_aug_data_samples)
        for pred in preds:
            self.assertEqual(pred.pred_text.text, 'cdefg')
