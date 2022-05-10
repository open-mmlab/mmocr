# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmocr.models.common.losses import DiceLoss
from mmocr.models.textrecog.losses import (ABILoss, CELoss, CTCLoss, SARLoss,
                                           TFLoss)


def test_ctc_loss():
    with pytest.raises(AssertionError):
        CTCLoss(flatten='flatten')
    with pytest.raises(AssertionError):
        CTCLoss(blank=None)
    with pytest.raises(AssertionError):
        CTCLoss(reduction=1)
    with pytest.raises(AssertionError):
        CTCLoss(zero_infinity='zero')
    # test CTCLoss
    ctc_loss = CTCLoss()
    outputs = torch.zeros(2, 40, 37)
    targets_dict = {
        'flatten_targets': torch.IntTensor([1, 2, 3, 4, 5]),
        'target_lengths': torch.LongTensor([2, 3])
    }

    losses = ctc_loss(outputs, targets_dict)
    assert isinstance(losses, dict)
    assert 'loss_ctc' in losses
    assert torch.allclose(losses['loss_ctc'],
                          torch.tensor(losses['loss_ctc'].item()).float())


def test_ce_loss():
    with pytest.raises(AssertionError):
        CELoss(ignore_index='ignore')
    with pytest.raises(AssertionError):
        CELoss(reduction=1)
    with pytest.raises(AssertionError):
        CELoss(reduction='avg')

    ce_loss = CELoss(ignore_index=0)
    outputs = torch.rand(1, 10, 37)
    targets_dict = {
        'padded_targets': torch.LongTensor([[1, 2, 3, 4, 0, 0, 0, 0, 0, 0]])
    }
    losses = ce_loss(outputs, targets_dict)
    assert isinstance(losses, dict)
    assert 'loss_ce' in losses
    assert losses['loss_ce'].size(1) == 10

    ce_loss = CELoss(ignore_first_char=True)
    outputs = torch.rand(1, 10, 37)
    targets_dict = {
        'padded_targets': torch.LongTensor([[1, 2, 3, 4, 0, 0, 0, 0, 0, 0]])
    }
    new_output, new_target = ce_loss.format(outputs, targets_dict)
    assert new_output.shape == torch.Size([1, 37, 9])
    assert new_target.shape == torch.Size([1, 9])


def test_sar_loss():
    outputs = torch.rand(1, 10, 37)
    targets_dict = {
        'padded_targets': torch.LongTensor([[1, 2, 3, 4, 0, 0, 0, 0, 0, 0]])
    }
    sar_loss = SARLoss()
    new_output, new_target = sar_loss.format(outputs, targets_dict)
    assert new_output.shape == torch.Size([1, 37, 9])
    assert new_target.shape == torch.Size([1, 9])


def test_tf_loss():
    with pytest.raises(AssertionError):
        TFLoss(flatten=1.0)

    outputs = torch.rand(1, 10, 37)
    targets_dict = {
        'padded_targets': torch.LongTensor([[1, 2, 3, 4, 0, 0, 0, 0, 0, 0]])
    }
    tf_loss = TFLoss(flatten=False)
    new_output, new_target = tf_loss.format(outputs, targets_dict)
    assert new_output.shape == torch.Size([1, 37, 9])
    assert new_target.shape == torch.Size([1, 9])


def test_dice_loss():
    with pytest.raises(AssertionError):
        DiceLoss(eps='1')

    dice_loss = DiceLoss()
    pred = torch.rand(1, 1, 32, 32)
    gt = torch.rand(1, 1, 32, 32)

    loss = dice_loss(pred, gt, None)
    assert isinstance(loss, torch.Tensor)

    mask = torch.rand(1, 1, 1, 1)
    loss = dice_loss(pred, gt, mask)
    assert isinstance(loss, torch.Tensor)


def test_abi_loss():
    loss = ABILoss(num_classes=90)
    outputs = dict(
        out_enc=dict(logits=torch.randn(2, 10, 90)),
        out_decs=[
            dict(logits=torch.randn(2, 10, 90)),
            dict(logits=torch.randn(2, 10, 90))
        ],
        out_fusers=[
            dict(logits=torch.randn(2, 10, 90)),
            dict(logits=torch.randn(2, 10, 90))
        ])
    targets_dict = {
        'padded_targets': torch.LongTensor([[1, 2, 3, 4, 0, 0, 0, 0, 0, 0]]),
        'targets':
        [torch.LongTensor([1, 2, 3, 4]),
         torch.LongTensor([1, 2, 3])]
    }
    result = loss(outputs, targets_dict)
    assert isinstance(result, dict)
    assert isinstance(result['loss_visual'], torch.Tensor)
    assert isinstance(result['loss_lang'], torch.Tensor)
    assert isinstance(result['loss_fusion'], torch.Tensor)

    outputs.pop('out_enc')
    loss(outputs, targets_dict)
    outputs.pop('out_decs')
    loss(outputs, targets_dict)
    outputs.pop('out_fusers')
    with pytest.raises(AssertionError):
        loss(outputs, targets_dict)
