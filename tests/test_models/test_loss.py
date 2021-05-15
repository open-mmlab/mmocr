import numpy as np
import torch

import mmocr.models.textdet.losses as losses
from mmdet.core import BitmapMasks


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
    fcenetloss = losses.FCELoss(fourier_degree=k, sample_num=10)

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

    # test balanced_bce_loss
    pred = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float)
    target = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.long)
    mask = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.long)
    bce_loss = drrgloss.balance_bce_loss(pred, target, mask).item()
    assert np.allclose(bce_loss, 0)

    # test gcn_loss
    preds = torch.tensor([[0., 1.], [1., 0.]])
    labels = torch.tensor([1, 0], dtype=torch.long)
    gcn_loss = drrgloss.gcn_loss((preds, labels))
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
