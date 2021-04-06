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
