# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmocr.models.textrecog.encoders import (ABIVisionModel, BaseEncoder,
                                             NRTREncoder, SAREncoder,
                                             SatrnEncoder, TransformerEncoder, ASTEREncoder)


def test_sar_encoder():
    with pytest.raises(AssertionError):
        SAREncoder(enc_bi_rnn='bi')
    with pytest.raises(AssertionError):
        SAREncoder(enc_do_rnn=2)
    with pytest.raises(AssertionError):
        SAREncoder(enc_gru='gru')
    with pytest.raises(AssertionError):
        SAREncoder(d_model=512.5)
    with pytest.raises(AssertionError):
        SAREncoder(d_enc=200.5)
    with pytest.raises(AssertionError):
        SAREncoder(mask='mask')

    encoder = SAREncoder()
    encoder.init_weights()
    encoder.train()

    feat = torch.randn(1, 512, 4, 40)
    img_metas = [{'valid_ratio': 1.0}]
    with pytest.raises(AssertionError):
        encoder(feat, img_metas * 2)
    out_enc = encoder(feat, img_metas)

    assert out_enc.shape == torch.Size([1, 512])


def test_nrtr_encoder():
    tf_encoder = NRTREncoder()
    tf_encoder.init_weights()
    tf_encoder.train()

    feat = torch.randn(1, 512, 1, 25)
    out_enc = tf_encoder(feat)
    print('hello', out_enc.size())
    assert out_enc.shape == torch.Size([1, 25, 512])


def test_satrn_encoder():
    satrn_encoder = SatrnEncoder()
    satrn_encoder.init_weights()
    satrn_encoder.train()

    feat = torch.randn(1, 512, 8, 25)
    out_enc = satrn_encoder(feat)
    assert out_enc.shape == torch.Size([1, 200, 512])


def test_base_encoder():
    encoder = BaseEncoder()
    encoder.init_weights()
    encoder.train()

    feat = torch.randn(1, 256, 4, 40)
    out_enc = encoder(feat)
    assert out_enc.shape == torch.Size([1, 256, 4, 40])


def test_transformer_encoder():
    model = TransformerEncoder()
    x = torch.randn(10, 512, 8, 32)
    assert model(x).shape == torch.Size([10, 512, 8, 32])


def test_abi_vision_model():
    model = ABIVisionModel(
        decoder=dict(type='ABIVisionDecoder', max_seq_len=10, use_result=None))
    x = torch.randn(1, 512, 8, 32)
    result = model(x)
    assert result['feature'].shape == torch.Size([1, 10, 512])
    assert result['logits'].shape == torch.Size([1, 10, 90])
    assert result['attn_scores'].shape == torch.Size([1, 10, 8, 32])

def test_aster_encoder():
    model = ASTEREncoder(in_channels=512, num_classes=512, with_lstm=True)
    feat = torch.randn(1, 512, 1, 25)
    assert model(feat).shape == torch.Size([1, 25, 512])
