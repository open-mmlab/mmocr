import pytest
import torch

from mmocr.models.textrecog import SegHead


def test_seg_head():
    with pytest.raises(AssertionError):
        SegHead(num_classes='100')
    with pytest.raises(AssertionError):
        SegHead(num_classes=-1)

    seg_head = SegHead(num_classes=37)
    out_neck = (torch.rand(1, 128, 32, 32), )
    out_head = seg_head(out_neck)
    assert out_head.shape == torch.Size([1, 37, 32, 32])
