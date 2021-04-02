import numpy as np
import pytest
import torch

from mmdet.core import BitmapMasks
from mmocr.models.common.losses import DiceLoss
from mmocr.models.textrecog.losses import (CAFCNLoss, CELoss, CTCLoss, SARLoss,
                                           TFLoss)


def test_ctc_loss():
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
    print(losses['loss_ce'].size())
    assert losses['loss_ce'].size(1) == 10


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


def test_cafcn_loss():
    with pytest.raises(AssertionError):
        CAFCNLoss(alpha='1')
    with pytest.raises(AssertionError):
        CAFCNLoss(attn_s2_downsample_ratio='2')
    with pytest.raises(AssertionError):
        CAFCNLoss(attn_s3_downsample_ratio='1.5')
    with pytest.raises(AssertionError):
        CAFCNLoss(seg_downsample_ratio='1.5')
    with pytest.raises(AssertionError):
        CAFCNLoss(attn_s2_downsample_ratio=2)
    with pytest.raises(AssertionError):
        CAFCNLoss(attn_s3_downsample_ratio=1.5)
    with pytest.raises(AssertionError):
        CAFCNLoss(seg_downsample_ratio=1.5)

    bsz = 1
    H = W = 64
    out_neck = (torch.ones(bsz, 1, H // 4, W // 4) * 0.5,
                torch.ones(bsz, 1, H // 8, W // 8) * 0.5,
                torch.ones(bsz, 1, H // 8, W // 8) * 0.5,
                torch.ones(bsz, 1, H // 8, W // 8) * 0.5,
                torch.ones(bsz, 1, H // 2, W // 2) * 0.5)
    out_head = torch.rand(bsz, 37, H // 2, W // 2)

    attn_tgt = np.zeros((H, W), dtype=np.float32)
    segm_tgt = np.zeros((H, W), dtype=np.float32)
    mask = np.ones((H, W), dtype=np.float32)
    gt_kernels = BitmapMasks([attn_tgt, segm_tgt, mask], H, W)

    cafcn_loss = CAFCNLoss()
    losses = cafcn_loss(out_neck, out_head, [gt_kernels])
    assert isinstance(losses, dict)
    assert 'loss_seg' in losses
    assert torch.allclose(losses['loss_seg'],
                          torch.tensor(losses['loss_seg'].item()).float())


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
