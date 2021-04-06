import pytest
import torch

from mmocr.models.textrecog.necks.cafcn_neck import (CAFCNNeck, CharAttn,
                                                     FeatGenerator)


def test_char_attn():
    with pytest.raises(AssertionError):
        CharAttn(in_channels=5.0)
    with pytest.raises(AssertionError):
        CharAttn(deformable='deformabel')

    in_feat = torch.rand(1, 128, 32, 32)
    char_attn = CharAttn()
    out_feat_map, attn_map = char_attn(in_feat)
    assert attn_map.shape == torch.Size([1, 1, 32, 32])
    assert out_feat_map.shape == torch.Size([1, 128, 32, 32])


def test_feat_generator():
    in_feat = torch.rand(1, 128, 32, 32)
    feat_generator = FeatGenerator(
        in_channels=128, out_channels=128, deformable=False)

    attn_map, feat_map = feat_generator(in_feat)
    assert attn_map.shape == torch.Size([1, 1, 32, 32])
    assert feat_map.shape == torch.Size([1, 128, 32, 32])


def test_cafcn_neck():
    in_s1 = torch.rand(1, 64, 64, 64)
    in_s2 = torch.rand(1, 128, 32, 32)
    in_s3 = torch.rand(1, 256, 16, 16)
    in_s4 = torch.rand(1, 512, 16, 16)
    in_s5 = torch.rand(1, 512, 16, 16)

    cafcn_neck = CAFCNNeck(deformable=False)
    cafcn_neck.init_weights()
    cafcn_neck.train()

    out_neck = cafcn_neck((in_s1, in_s2, in_s3, in_s4, in_s5))
    assert out_neck[0].shape == torch.Size([1, 1, 32, 32])
    assert out_neck[1].shape == torch.Size([1, 1, 16, 16])
    assert out_neck[2].shape == torch.Size([1, 1, 16, 16])
    assert out_neck[3].shape == torch.Size([1, 1, 16, 16])
    assert out_neck[4].shape == torch.Size([1, 128, 64, 64])
