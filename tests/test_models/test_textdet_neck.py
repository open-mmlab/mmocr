import torch

from mmocr.models.textdet.necks import FPNC


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
