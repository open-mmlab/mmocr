# Copyright (c) OpenMMLab. All rights reserved.
import math

import pytest
import torch

from mmocr.models.textrecog.decoders import (ABILanguageDecoder,
                                             ABIVisionDecoder, BaseDecoder,
                                             NRTRDecoder, ParallelSARDecoder,
                                             ParallelSARDecoderWithBS,
                                             SequentialSARDecoder,
                                             ASTERDecoder)
from mmocr.models.textrecog.decoders.sar_decoder_with_bs import DecodeNode


def _create_dummy_input():
    feat = torch.rand(1, 512, 4, 40)
    out_enc = torch.rand(1, 512)
    tgt_dict = {'padded_targets': torch.LongTensor([[1, 1, 1, 1, 36]])}
    img_metas = [{'valid_ratio': 1.0}]

    return feat, out_enc, tgt_dict, img_metas


def test_base_decoder():
    decoder = BaseDecoder()
    with pytest.raises(NotImplementedError):
        decoder.forward_train(None, None, None, None)
    with pytest.raises(NotImplementedError):
        decoder.forward_test(None, None, None)


def test_parallel_sar_decoder():
    # test parallel sar decoder
    decoder = ParallelSARDecoder(num_classes=37, padding_idx=36, max_seq_len=5)
    decoder.init_weights()
    decoder.train()

    feat, out_enc, tgt_dict, img_metas = _create_dummy_input()
    with pytest.raises(AssertionError):
        decoder(feat, out_enc, tgt_dict, [], True)
    with pytest.raises(AssertionError):
        decoder(feat, out_enc, tgt_dict, img_metas * 2, True)

    out_train = decoder(feat, out_enc, tgt_dict, img_metas, True)
    assert out_train.shape == torch.Size([1, 5, 36])

    out_test = decoder(feat, out_enc, tgt_dict, img_metas, False)
    assert out_test.shape == torch.Size([1, 5, 36])


def test_sequential_sar_decoder():
    # test parallel sar decoder
    decoder = SequentialSARDecoder(
        num_classes=37, padding_idx=36, max_seq_len=5)
    decoder.init_weights()

def test_aster_decoder():
    model = ASTERDecoder(in_planes=512, num_classes=97, s_Dim=512, Atten_Dim=512)
    x = torch.randn(1, 25, 512)
    result = model(x)
    assert result == torch.Size([1, 512])

    decoder.train()

    feat, out_enc, tgt_dict, img_metas = _create_dummy_input()
    with pytest.raises(AssertionError):
        decoder(feat, out_enc, tgt_dict, [])
    with pytest.raises(AssertionError):
        decoder(feat, out_enc, tgt_dict, img_metas * 2)

    out_train = decoder(feat, out_enc, tgt_dict, img_metas, True)
    assert out_train.shape == torch.Size([1, 5, 36])

    out_test = decoder(feat, out_enc, tgt_dict, img_metas, False)
    assert out_test.shape == torch.Size([1, 5, 36])


def test_parallel_sar_decoder_with_beam_search():
    with pytest.raises(AssertionError):
        ParallelSARDecoderWithBS(beam_width='beam')
    with pytest.raises(AssertionError):
        ParallelSARDecoderWithBS(beam_width=0)

    feat, out_enc, tgt_dict, img_metas = _create_dummy_input()
    decoder = ParallelSARDecoderWithBS(
        beam_width=1, num_classes=37, padding_idx=36, max_seq_len=5)
    decoder.init_weights()
    decoder.train()
    with pytest.raises(AssertionError):
        decoder(feat, out_enc, tgt_dict, [])
    with pytest.raises(AssertionError):
        decoder(feat, out_enc, tgt_dict, img_metas * 2)

    out_test = decoder(feat, out_enc, tgt_dict, img_metas, train_mode=False)
    assert out_test.shape == torch.Size([1, 5, 36])

    # test decodenode
    with pytest.raises(AssertionError):
        DecodeNode(1, 1)
    with pytest.raises(AssertionError):
        DecodeNode([1, 2], ['4', '3'])
    with pytest.raises(AssertionError):
        DecodeNode([1, 2], [0.5])
    decode_node = DecodeNode([1, 2], [0.7, 0.8])
    assert math.isclose(decode_node.eval(), 1.5)


def test_transformer_decoder():
    decoder = NRTRDecoder(num_classes=37, padding_idx=36, max_seq_len=5)
    decoder.init_weights()
    decoder.train()

    out_enc = torch.rand(1, 25, 512)
    tgt_dict = {'padded_targets': torch.LongTensor([[1, 1, 1, 1, 36]])}
    img_metas = [{'valid_ratio': 1.0}]
    tgt_dict['padded_targets'] = tgt_dict['padded_targets']

    out_train = decoder(None, out_enc, tgt_dict, img_metas, True)
    assert out_train.shape == torch.Size([1, 5, 36])

    out_test = decoder(None, out_enc, tgt_dict, img_metas, False)
    assert out_test.shape == torch.Size([1, 5, 36])


def test_abi_language_decoder():
    decoder = ABILanguageDecoder(max_seq_len=25)
    logits = torch.randn(2, 25, 90)
    result = decoder(
        feat=None, out_enc=logits, targets_dict=None, img_metas=None)
    assert result['feature'].shape == torch.Size([2, 25, 512])
    assert result['logits'].shape == torch.Size([2, 25, 90])


def test_abi_vision_decoder():
    model = ABIVisionDecoder(
        in_channels=128, num_channels=16, max_seq_len=10, use_result=None)
    x = torch.randn(2, 128, 8, 32)
    result = model(x, None)
    assert result['feature'].shape == torch.Size([2, 10, 128])
    assert result['logits'].shape == torch.Size([2, 10, 90])
    assert result['attn_scores'].shape == torch.Size([2, 10, 8, 32])
