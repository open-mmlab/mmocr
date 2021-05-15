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
