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

    k = 1
    preds = []
    preds.append(torch.ones(1, 4, 10, 10))
    preds.append(torch.ones(1, 4 * k + 2, 10, 10))

    boundaries = fcenet_decode(
        preds=preds,
        fourier_degree=k,
        num_reconstr_points=50,
        scale=1,
        nms_thr=0.01)

    assert isinstance(boundaries, list)


def test_poly_nms():
    from mmocr.models.textdet.postprocess.wrapper import poly_nms
    threshold = 0
    polygons = []
    polygons.append([10, 10, 10, 30, 30, 30, 30, 10, 0.95])
    polygons.append([15, 15, 15, 25, 25, 25, 25, 15, 0.9])
    polygons.append([40, 40, 40, 50, 50, 50, 50, 40, 0.85])
    polygons.append([5, 5, 5, 15, 15, 15, 15, 5, 0.7])

    keep_poly = poly_nms(polygons, threshold)
    assert isinstance(keep_poly, list)


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
