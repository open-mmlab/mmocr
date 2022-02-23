# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmocr.core.bbox import Mixer


def test_mixer():
    img_metas = dict(
        bboxes=torch.FloatTensor([[0, 0, 0, 1, 1, 0, 1, 1],
                                  [1, 0, 1, 1, 2, 0, 2, 1]]),
        texts=['hello', 'world'],
    )
    pred_results = dict(
        bboxes=torch.FloatTensor(
            [[0.5, 0.5, 0, 1, 1, 0, 1, 1], [-1, -1, -1, -1, -1, -1, -1, 1],
             [-1, -1, -1, -1, -1, -1, -1, 1], [-0.5, -0.5, 0, 1, 1, 0, 1, 1],
             [1, 0, 1, 1, 2, 0, 2, 1]]))
    gt_inds = torch.LongTensor([1, 0, -1, 1, 2])

    mixer = Mixer(pred_keys=['bboxes'], add_gt=True)
    new_img_metas = mixer(img_metas, pred_results, gt_inds)
    target_bboxes = torch.FloatTensor([[0.5, 0.5, 0, 1, 1, 0, 1, 1],
                                       [-0.5, -0.5, 0, 1, 1, 0, 1, 1],
                                       [1, 0, 1, 1, 2, 0, 2, 1],
                                       [0, 0, 0, 1, 1, 0, 1, 1],
                                       [1, 0, 1, 1, 2, 0, 2, 1]])
    target_texts = ['hello', 'hello', 'world', 'hello', 'world']
    assert torch.allclose(new_img_metas['bboxes'], target_bboxes)
    assert target_texts == new_img_metas['texts']

    mixer = Mixer(pred_keys=['bboxes'], add_gt=False)
    gt_inds = torch.LongTensor([-1, -1, 2, 2, 1])
    new_img_metas = mixer(img_metas, pred_results, gt_inds)
    target_bboxes = torch.FloatTensor([[-1, -1, -1, -1, -1, -1, -1, 1],
                                       [-0.5, -0.5, 0, 1, 1, 0, 1, 1],
                                       [1, 0, 1, 1, 2, 0, 2, 1]])
    target_texts = ['world', 'world', 'hello']
    assert torch.allclose(new_img_metas['bboxes'], target_bboxes)
    assert target_texts == new_img_metas['texts']
