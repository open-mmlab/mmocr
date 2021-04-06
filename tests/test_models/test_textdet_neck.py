import pytest
import torch

from mmocr.models.textdet.necks import FPN_UNET, FPNC


def test_fpnc():

    in_channels = [64, 128, 256, 512]
    size = [112, 56, 28, 14]
    for flag in [False, True]:
        fpnc = FPNC(
            in_channels=in_channels,
            bias_on_lateral=flag,
            bn_re_on_lateral=flag,
            bias_on_smooth=flag,
            bn_re_on_smooth=flag,
            conv_after_concat=flag)
        fpnc.init_weights()
        inputs = []
        for i in range(4):
            inputs.append(torch.rand(1, in_channels[i], size[i], size[i]))
        outputs = fpnc.forward(inputs)
        assert list(outputs.size()) == [1, 256, 112, 112]


def test_fpn_unet_neck():
    s = 64
    feat_sizes = [s // 2**i for i in range(4)]  # [64, 32, 16, 8]
    in_channels = [8, 16, 32, 64]
    out_channels = 4

    # len(in_channcels) is not equal to 4
    with pytest.raises(AssertionError):
        FPN_UNET(in_channels + [128], out_channels)

    # `out_channels` is not int type
    with pytest.raises(AssertionError):
        FPN_UNET(in_channels, [2, 4])

    feats = [
        torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i])
        for i in range(len(in_channels))
    ]

    fpn_unet_neck = FPN_UNET(in_channels, out_channels)
    fpn_unet_neck.init_weights()

    out_neck = fpn_unet_neck(feats)
    assert out_neck.shape == torch.Size([1, out_channels, s * 4, s * 4])
