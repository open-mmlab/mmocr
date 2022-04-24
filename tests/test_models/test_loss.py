# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
from mmdet.core import BitmapMasks

import mmocr.models.textdet.losses as losses
from mmocr.models.textdet.dense_heads import FCOSHead


def test_panloss():
    panloss = losses.PANLoss()

    # test bitmasks2tensor
    mask = [[1, 0, 1], [1, 1, 1], [0, 0, 1]]
    target = [[1, 0, 1, 0, 0], [1, 1, 1, 0, 0], [0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    masks = [np.array(mask)]
    bitmasks = BitmapMasks(masks, 3, 3)
    target_sz = (6, 5)
    results = panloss.bitmasks2tensor([bitmasks], target_sz)
    assert len(results) == 1
    assert torch.sum(torch.abs(results[0].float() -
                               torch.Tensor(target))).item() == 0


def test_textsnakeloss():
    textsnakeloss = losses.TextSnakeLoss()

    # test balanced_bce_loss
    pred = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float)
    target = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.long)
    mask = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.long)
    bce_loss = textsnakeloss.balanced_bce_loss(pred, target, mask).item()

    assert np.allclose(bce_loss, 0)


def test_fcenetloss():
    k = 5
    fcenetloss = losses.FCELoss(fourier_degree=k, num_sample=10)

    input_shape = (1, 3, 64, 64)
    (n, c, h, w) = input_shape

    # test ohem
    pred = torch.ones((200, 2), dtype=torch.float)
    target = torch.ones(200, dtype=torch.long)
    target[20:] = 0
    mask = torch.ones(200, dtype=torch.long)

    ohem_loss1 = fcenetloss.ohem(pred, target, mask)
    ohem_loss2 = fcenetloss.ohem(pred, target, 1 - mask)
    assert isinstance(ohem_loss1, torch.Tensor)
    assert isinstance(ohem_loss2, torch.Tensor)

    # test forward
    preds = []
    for i in range(n):
        scale = 8 * 2**i
        pred = []
        pred.append(torch.rand(n, 4, h // scale, w // scale))
        pred.append(torch.rand(n, 4 * k + 2, h // scale, w // scale))
        preds.append(pred)

    p3_maps = []
    p4_maps = []
    p5_maps = []
    for _ in range(n):
        p3_maps.append(np.random.random((5 + 4 * k, h // 8, w // 8)))
        p4_maps.append(np.random.random((5 + 4 * k, h // 16, w // 16)))
        p5_maps.append(np.random.random((5 + 4 * k, h // 32, w // 32)))

    loss = fcenetloss(preds, 0, p3_maps, p4_maps, p5_maps)
    assert isinstance(loss, dict)


def test_drrgloss():
    drrgloss = losses.DRRGLoss()
    assert np.allclose(drrgloss.ohem_ratio, 3.0)

    # test balance_bce_loss
    pred = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float)
    target = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.long)
    mask = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.long)
    bce_loss = drrgloss.balance_bce_loss(pred, target, mask).item()
    assert np.allclose(bce_loss, 0)

    # test balance_bce_loss with positive_count equal to zero
    pred = torch.ones((16, 16), dtype=torch.float)
    target = torch.ones((16, 16), dtype=torch.long)
    mask = torch.zeros((16, 16), dtype=torch.long)
    bce_loss = drrgloss.balance_bce_loss(pred, target, mask).item()
    assert np.allclose(bce_loss, 0)

    # test gcn_loss
    gcn_preds = torch.tensor([[0., 1.], [1., 0.]])
    labels = torch.tensor([1, 0], dtype=torch.long)
    gcn_loss = drrgloss.gcn_loss((gcn_preds, labels))
    assert gcn_loss.item()

    # test bitmasks2tensor
    mask = [[1, 0, 1], [1, 1, 1], [0, 0, 1]]
    target = [[1, 0, 1, 0, 0], [1, 1, 1, 0, 0], [0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    masks = [np.array(mask)]
    bitmasks = BitmapMasks(masks, 3, 3)
    target_sz = (6, 5)
    results = drrgloss.bitmasks2tensor([bitmasks], target_sz)
    assert len(results) == 1
    assert torch.sum(torch.abs(results[0].float() -
                               torch.Tensor(target))).item() == 0

    # test forward
    target_maps = [BitmapMasks([np.random.randn(20, 20)], 20, 20)]
    target_masks = [BitmapMasks([np.ones((20, 20))], 20, 20)]
    gt_masks = [BitmapMasks([np.ones((20, 20))], 20, 20)]
    preds = (torch.randn((1, 6, 20, 20)), (gcn_preds, labels))
    loss_dict = drrgloss(preds, 1., target_masks, target_masks, gt_masks,
                         target_maps, target_maps, target_maps, target_maps)

    assert isinstance(loss_dict, dict)
    assert 'loss_text' in loss_dict.keys()
    assert 'loss_center' in loss_dict.keys()
    assert 'loss_height' in loss_dict.keys()
    assert 'loss_sin' in loss_dict.keys()
    assert 'loss_cos' in loss_dict.keys()
    assert 'loss_gcn' in loss_dict.keys()

    # test forward with downsample_ratio less than 1.
    target_maps = [BitmapMasks([np.random.randn(40, 40)], 40, 40)]
    target_masks = [BitmapMasks([np.ones((40, 40))], 40, 40)]
    gt_masks = [BitmapMasks([np.ones((40, 40))], 40, 40)]
    preds = (torch.randn((1, 6, 20, 20)), (gcn_preds, labels))
    loss_dict = drrgloss(preds, 0.5, target_masks, target_masks, gt_masks,
                         target_maps, target_maps, target_maps, target_maps)

    assert isinstance(loss_dict, dict)

    # test forward with blank gt_mask.
    target_maps = [BitmapMasks([np.random.randn(20, 20)], 20, 20)]
    target_masks = [BitmapMasks([np.ones((20, 20))], 20, 20)]
    gt_masks = [BitmapMasks([np.zeros((20, 20))], 20, 20)]
    preds = (torch.randn((1, 6, 20, 20)), (gcn_preds, labels))
    loss_dict = drrgloss(preds, 1., target_masks, target_masks, gt_masks,
                         target_maps, target_maps, target_maps, target_maps)

    assert isinstance(loss_dict, dict)


def test_dice_loss():
    pred = torch.Tensor([[[-1000, -1000, -1000], [-1000, -1000, -1000],
                          [-1000, -1000, -1000]]])
    target = torch.Tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    mask = torch.Tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]])

    pan_loss = losses.PANLoss()

    dice_loss = pan_loss.dice_loss_with_logits(pred, target, mask)

    assert np.allclose(dice_loss.item(), 0)


@pytest.mark.parametrize('with_bezier', [True, False])
def test_fcos_loss(with_bezier):
    """Tests fcos head loss when truth is empty and non-empty."""
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]
    head = FCOSHead(num_classes=4, in_channels=1, with_bezier=with_bezier)
    # Focal Loss is not supported on CPU
    loss = losses.FCOSLoss(
        num_classes=4,
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        with_bezier=with_bezier)
    feat = [
        torch.rand(1, 1, s // feat_size, s // feat_size)
        for feat_size in [4, 8, 16, 32, 64]
    ]
    preds = head.forward(feat)
    # Test that empty ground truth encourages the network to predict background
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]
    gt_bboxes_ignore = None
    if with_bezier:
        gt_beziers = [torch.empty((0, 16))]
        empty_gt_losses = loss(preds, img_metas, gt_bboxes, gt_labels,
                               gt_bboxes_ignore, gt_beziers)
    else:
        empty_gt_losses = loss(preds, img_metas, gt_bboxes, gt_labels,
                               gt_bboxes_ignore)
    # When there is no truth, the cls loss should be nonzero but there should
    # be no box loss.
    empty_cls_loss = empty_gt_losses['loss_cls']
    empty_box_loss = empty_gt_losses['loss_bbox']
    assert empty_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert empty_box_loss.item() == 0, (
        'there should be no box loss when there are no true boxes')
    if with_bezier:
        empty_bezier_loss = empty_gt_losses['loss_bezier']
        assert empty_bezier_loss.item() == 0, 'bezier loss should be zero'

    # When truth is non-empty then both cls and box loss should be nonzero for
    # random inputs
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    if with_bezier:
        gt_beziers = [torch.randn(1, 16)]
        one_gt_losses = loss(preds, img_metas, gt_bboxes, gt_labels,
                             gt_bboxes_ignore, gt_beziers)
    else:
        one_gt_losses = loss(preds, img_metas, gt_bboxes, gt_labels,
                             gt_bboxes_ignore)
    onegt_cls_loss = one_gt_losses['loss_cls']
    onegt_box_loss = one_gt_losses['loss_bbox']
    assert onegt_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert onegt_box_loss.item() > 0, 'box loss should be non-zero'
    if with_bezier:
        one_bezier_loss = one_gt_losses['loss_bezier']
        assert one_bezier_loss.item() > 0, 'bezier loss should be non-zero'
