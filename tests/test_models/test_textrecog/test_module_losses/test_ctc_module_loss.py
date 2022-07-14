# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

import torch
from mmengine.data import LabelData

from mmocr.data import TextRecogDataSample
from mmocr.models.textrecog.dictionary import Dictionary
from mmocr.models.textrecog.module_losses import CTCModuleLoss
from mmocr.testing import create_dummy_dict_file


class TestCTCModuleLoss(TestCase):

    def test_ctc_loss(self):
        tmp_dir = tempfile.TemporaryDirectory()
        # create dummy data
        dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
        create_dummy_dict_file(dict_file)

        dictionary = Dictionary(dict_file=dict_file, with_padding=True)
        with self.assertRaises(AssertionError):
            CTCModuleLoss(dictionary=dictionary, flatten='flatten')
        with self.assertRaises(AssertionError):
            CTCModuleLoss(dictionary=dictionary, reduction=1)
        with self.assertRaises(AssertionError):
            CTCModuleLoss(dictionary=dictionary, zero_infinity='zero')

        outputs = torch.zeros(2, 40, 37)
        datasample1 = TextRecogDataSample()
        gt_text1 = LabelData(item='hell')
        datasample1.gt_text = gt_text1
        datasample2 = datasample1.clone()
        gt_text2 = LabelData(item='owrd')
        datasample2.gt_text = gt_text2
        data_samples = [datasample1, datasample2]
        ctc_loss = CTCModuleLoss(dictionary=dictionary)
        data_samples = ctc_loss.get_targets(data_samples)
        losses = ctc_loss(outputs, data_samples)
        assert isinstance(losses, dict)
        assert 'loss_ctc' in losses
        assert torch.allclose(losses['loss_ctc'],
                              torch.tensor(losses['loss_ctc'].item()).float())
        # test flatten = False
        ctc_loss = CTCModuleLoss(dictionary=dictionary, flatten=False)
        losses = ctc_loss(outputs, data_samples)
        assert isinstance(losses, dict)
        assert 'loss_ctc' in losses
        assert torch.allclose(losses['loss_ctc'],
                              torch.tensor(losses['loss_ctc'].item()).float())
        tmp_dir.cleanup()

    def test_get_targets(self):
        tmp_dir = tempfile.TemporaryDirectory()
        # create dummy data
        dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
        create_dummy_dict_file(dict_file, list('helowrd'))

        dictionary = Dictionary(dict_file=dict_file, with_padding=True)
        loss = CTCModuleLoss(dictionary=dictionary, letter_case='lower')
        # test encode str to tensor
        datasample1 = TextRecogDataSample()
        gt_text1 = LabelData(item='hell')
        datasample1.gt_text = gt_text1
        datasample2 = datasample1.clone()
        gt_text2 = LabelData(item='owrd')
        datasample2.gt_text = gt_text2

        data_samples = [datasample1, datasample2]
        expect_tensor1 = torch.IntTensor([0, 1, 2, 2])
        expect_tensor2 = torch.IntTensor([3, 4, 5, 6])

        data_samples = loss.get_targets(data_samples)
        self.assertTrue(
            torch.allclose(data_samples[0].gt_text.indexes, expect_tensor1))
        self.assertTrue(
            torch.allclose(data_samples[1].gt_text.indexes, expect_tensor2))
        tmp_dir.cleanup()
