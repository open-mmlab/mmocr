# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmocr.models.common import (PositionalEncoding, TFDecoderLayer,
                                 TFEncoderLayer)
from mmocr.models.textrecog.layers import BasicBlock, Bottleneck
from mmocr.models.textrecog.layers.attention_aster_layer import AttentionLSTM
from mmocr.models.textrecog.layers.conv_layer import conv3x3


def test_conv_layer():
    conv3by3 = conv3x3(3, 6)
    assert conv3by3.in_channels == 3
    assert conv3by3.out_channels == 6
    assert conv3by3.kernel_size == (3, 3)

    x = torch.rand(1, 64, 224, 224)
    # test basic block
    basic_block = BasicBlock(64, 64)
    assert basic_block.expansion == 1

    out = basic_block(x)

    assert out.shape == torch.Size([1, 64, 224, 224])

    # test bottle neck
    bottle_neck = Bottleneck(64, 64, downsample=True)
    assert bottle_neck.expansion == 4

    out = bottle_neck(x)

    assert out.shape == torch.Size([1, 256, 224, 224])


def test_transformer_layer():
    # test decoder_layer
    decoder_layer = TFDecoderLayer()
    in_dec = torch.rand(1, 30, 512)
    out_enc = torch.rand(1, 128, 512)
    out_dec = decoder_layer(in_dec, out_enc)
    assert out_dec.shape == torch.Size([1, 30, 512])

    decoder_layer = TFDecoderLayer(
        operation_order=('self_attn', 'norm', 'enc_dec_attn', 'norm', 'ffn',
                         'norm'))
    out_dec = decoder_layer(in_dec, out_enc)
    assert out_dec.shape == torch.Size([1, 30, 512])

    # test positional_encoding
    pos_encoder = PositionalEncoding()
    x = torch.rand(1, 30, 512)
    out = pos_encoder(x)
    assert out.size() == x.size()

    # test encoder_layer
    encoder_layer = TFEncoderLayer()
    in_enc = torch.rand(1, 20, 512)
    out_enc = encoder_layer(in_enc)
    assert out_dec.shape == torch.Size([1, 30, 512])

    encoder_layer = TFEncoderLayer(
        operation_order=('self_attn', 'norm', 'ffn', 'norm'))
    out_enc = encoder_layer(in_enc)
    assert out_dec.shape == torch.Size([1, 30, 512])


def test_attentionLSTM_layer():
    decoder_layer = AttentionLSTM(H_Dim=512, S_Dim=512, Atten_Dim=512)
    x = torch.rand(1, 25, 512)
    state = torch.zeros(1, 1, 512)

    out_enc = decoder_layer(x, state)

    assert out_enc.shape == torch.Size([1, 25])
