# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmocr.models.textdet.necks import FPN_UNet


def test_fpn_unet_neck():
    s = 64
    feat_sizes = [s // 2**i for i in range(4)]  # [64, 32, 16, 8]
    in_channels = [8, 16, 32, 64]
    out_channels = 4

    # len(in_channcels) is not equal to 4
    with pytest.raises(AssertionError):
        FPN_UNet(in_channels + [128], out_channels)

    # `out_channels` is not int type
    with pytest.raises(AssertionError):
        FPN_UNet(in_channels, [2, 4])

    feats = [
        torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i])
        for i in range(len(in_channels))
    ]

    fpn_unet_neck = FPN_UNet(in_channels, out_channels)
    fpn_unet_neck.init_weights()

    out_neck = fpn_unet_neck(feats)
    assert out_neck.shape == torch.Size([1, out_channels, s * 4, s * 4])
