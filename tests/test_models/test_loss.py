import numpy as np
import torch
from mmdet.core import BitmapMasks

import mmocr.models.textdet.losses as losses


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
