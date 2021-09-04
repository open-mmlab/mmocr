# Copyright (c) OpenMMLab. All rights reserved.
import math

import pytest
import torch

from mmocr.models.textrecog.decoders import (BaseDecoder, ParallelSARDecoder,
                                             ParallelSARDecoderWithBS,
                                             SequentialSARDecoder, TFDecoder)
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
    decoder = TFDecoder(num_classes=37, padding_idx=36, max_seq_len=5)
    decoder.init_weights()
    decoder.train()

    out_enc = torch.rand(1, 512, 1, 25)
    tgt_dict = {'padded_targets': torch.LongTensor([[1, 1, 1, 1, 36]])}
    img_metas = [{'valid_ratio': 1.0}]
    tgt_dict['padded_targets'] = tgt_dict['padded_targets']

    out_train = decoder(None, out_enc, tgt_dict, img_metas, True)
    assert out_train.shape == torch.Size([1, 5, 36])

    out_test = decoder(None, out_enc, tgt_dict, img_metas, False)
    assert out_test.shape == torch.Size([1, 5, 36])
