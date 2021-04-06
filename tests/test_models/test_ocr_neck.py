import torch

from mmocr.models.textrecog.necks import FPNOCR


def test_fpn_ocr():
    in_s1 = torch.rand(1, 128, 32, 256)
    in_s2 = torch.rand(1, 256, 16, 128)
    in_s3 = torch.rand(1, 512, 8, 64)
    in_s4 = torch.rand(1, 512, 4, 32)

    fpn_ocr = FPNOCR(in_channels=[128, 256, 512, 512], out_channels=256)
    fpn_ocr.init_weights()
    fpn_ocr.train()

    out_neck = fpn_ocr((in_s1, in_s2, in_s3, in_s4))
    assert out_neck[0].shape == torch.Size([1, 256, 32, 256])
