import pytest
import torch

from mmocr.models.textspotter.roi_extractors import BezierRoIExtractor


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Need cuda')
def test_bezier_roi_extractor():

    cfg = dict(
        roi_layer=dict(
            type='BezierAlign',
            output_size=(8, 32),
        ),
        out_channels=16,
        featmap_strides=[4, 8, 16],
        finest_scale=10,
    )

    bezier_roi_extractor = BezierRoIExtractor(**cfg)

    feats = (torch.zeros((2, 16, 224, 224), dtype=torch.double),
             torch.zeros((2, 16, 112, 112), dtype=torch.double) + 1.,
             torch.zeros((2, 16, 56, 56), dtype=torch.double) + 2.)

    rois = torch.Tensor(
        [[0, 0, 0, 0, 1, 0, 2, 0, 3, 1, 0, 1, 1, 1, 2, 1, 3],
         [0, 0, 0, 0, 1, 0, 2, 0, 3, 20, 0, 20, 1, 20, 2, 20, 3],
         [1, 0, 0, 0, 1, 0, 2, 0, 3, 40, 0, 40, 1, 40, 2, 40, 3]]).double()
    feats = tuple(feat.to('cuda:0') for feat in feats)
    rois = rois.to('cuda:0')
    res = bezier_roi_extractor(feats, rois)
    assert res.shape == torch.Size([3, 16, 8, 32])
    assert torch.all(res[0, ...] == 0)
    assert torch.all(res[1, ...] == 1)
    assert torch.all(res[2, ...] == 2)
