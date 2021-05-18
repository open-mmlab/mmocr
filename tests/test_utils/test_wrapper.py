import numpy as np
import torch


def test_db_boxes_from_bitmaps():
    """Test the boxes_from_bitmaps function in db_decoder."""
    from mmocr.models.textdet.postprocess.wrapper import db_decode
    pred = np.array([[[0.8, 0.8, 0.8, 0.8, 0], [0.8, 0.8, 0.8, 0.8, 0],
                      [0.8, 0.8, 0.8, 0.8, 0], [0.8, 0.8, 0.8, 0.8, 0],
                      [0.8, 0.8, 0.8, 0.8, 0]]])
    preds = torch.FloatTensor(pred).requires_grad_(True)

    boxes = db_decode(preds, text_repr_type='quad', min_text_width=0)
    assert len(boxes) == 1


def test_fcenet_decode():
    from mmocr.models.textdet.postprocess.wrapper import fcenet_decode

    k = 5
    preds = []
    preds.append(torch.ones(1, 4, 40, 40))
    preds.append(torch.ones(1, 4 * k + 2, 40, 40))

    boundaries = fcenet_decode(
        preds=preds, fourier_degree=k, num_reconstr_points=50, scale=1)

    assert isinstance(boundaries, list)


def test_comps2boundaries():
    from mmocr.models.textdet.postprocess.wrapper import comps2boundaries

    # test comps2boundaries
    x1 = np.arange(2, 18, 2)
    x2 = x1 + 2
    y1 = np.ones(8) * 2
    y2 = y1 + 2
    comp_scores = np.ones(8, dtype=np.float32) * 0.9
    text_comps = np.stack([x1, y1, x2, y1, x2, y2, x1, y2,
                           comp_scores]).transpose()
    comp_labels = np.array([1, 1, 1, 1, 1, 3, 5, 5])
    shuffle = [3, 2, 5, 7, 6, 0, 4, 1]
    boundaries = comps2boundaries(text_comps[shuffle], comp_labels[shuffle])
    assert len(boundaries) == 3

    # test comps2boundaries with blank inputs
    boundaries = comps2boundaries(text_comps[[]], comp_labels[[]])
    assert len(boundaries) == 0
