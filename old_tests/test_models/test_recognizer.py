# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from functools import partial

import numpy as np
import pytest
import torch
from mmdet.core import BitmapMasks

from mmocr.models.textrecog.recognizers import (EncodeDecodeRecognizer,
                                                SegRecognizer)


def _create_dummy_dict_file(dict_file):
    chars = list('helowrd')
    with open(dict_file, 'w') as fw:
        for char in chars:
            fw.write(char + '\n')


def test_base_recognizer():
    tmp_dir = tempfile.TemporaryDirectory()
    # create dummy data
    dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
    _create_dummy_dict_file(dict_file)

    label_convertor = dict(
        type='CTCConvertor', dict_file=dict_file, with_unknown=False)

    preprocessor = None
    backbone = dict(type='VeryDeepVgg', leaky_relu=False)
    encoder = None
    decoder = dict(type='CRNNDecoder', in_channels=512, rnn_flag=True)
    loss = dict(type='CTCLoss')

    with pytest.raises(AssertionError):
        EncodeDecodeRecognizer(backbone=None)
    with pytest.raises(AssertionError):
        EncodeDecodeRecognizer(decoder=None)
    with pytest.raises(AssertionError):
        EncodeDecodeRecognizer(loss=None)
    with pytest.raises(AssertionError):
        EncodeDecodeRecognizer(label_convertor=None)

    recognizer = EncodeDecodeRecognizer(
        preprocessor=preprocessor,
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        loss=loss,
        label_convertor=label_convertor)

    recognizer.init_weights()
    recognizer.train()

    imgs = torch.rand(1, 3, 32, 160)

    # test extract feat
    feat = recognizer.extract_feat(imgs)
    assert feat.shape == torch.Size([1, 512, 1, 41])

    # test forward train
    img_metas = [{
        'text': 'hello',
        'resize_shape': (32, 120, 3),
        'valid_ratio': 1.0
    }]
    losses = recognizer.forward_train(imgs, img_metas)
    assert isinstance(losses, dict)
    assert 'loss_ctc' in losses

    # test simple test
    results = recognizer.simple_test(imgs, img_metas)
    assert isinstance(results, list)
    assert isinstance(results[0], dict)
    assert 'text' in results[0]
    assert 'score' in results[0]

    # test onnx export
    recognizer.forward = partial(
        recognizer.simple_test,
        img_metas=img_metas,
        return_loss=False,
        rescale=True)
    with tempfile.TemporaryDirectory() as tmpdirname:
        onnx_path = f'{tmpdirname}/tmp.onnx'
        torch.onnx.export(
            recognizer, (imgs, ),
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            export_params=True,
            keep_initializers_as_inputs=False)

    # test aug_test
    aug_results = recognizer.aug_test([imgs, imgs], [img_metas, img_metas])
    assert isinstance(aug_results, list)
    assert isinstance(aug_results[0], dict)
    assert 'text' in aug_results[0]
    assert 'score' in aug_results[0]

    tmp_dir.cleanup()


def test_seg_recognizer():
    tmp_dir = tempfile.TemporaryDirectory()
    # create dummy data
    dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
    _create_dummy_dict_file(dict_file)

    label_convertor = dict(
        type='SegConvertor', dict_file=dict_file, with_unknown=False)

    preprocessor = None
    backbone = dict(
        type='ResNet31OCR',
        layers=[1, 2, 5, 3],
        channels=[32, 64, 128, 256, 512, 512],
        out_indices=[0, 1, 2, 3],
        stage4_pool_cfg=dict(kernel_size=2, stride=2),
        last_stage_pool=True)
    neck = dict(
        type='FPNOCR', in_channels=[128, 256, 512, 512], out_channels=256)
    head = dict(
        type='SegHead',
        in_channels=256,
        upsample_param=dict(scale_factor=2.0, mode='nearest'))
    loss = dict(type='SegLoss', seg_downsample_ratio=1.0)

    with pytest.raises(AssertionError):
        SegRecognizer(backbone=None)
    with pytest.raises(AssertionError):
        SegRecognizer(neck=None)
    with pytest.raises(AssertionError):
        SegRecognizer(head=None)
    with pytest.raises(AssertionError):
        SegRecognizer(loss=None)
    with pytest.raises(AssertionError):
        SegRecognizer(label_convertor=None)

    recognizer = SegRecognizer(
        preprocessor=preprocessor,
        backbone=backbone,
        neck=neck,
        head=head,
        loss=loss,
        label_convertor=label_convertor)

    recognizer.init_weights()
    recognizer.train()

    imgs = torch.rand(1, 3, 64, 256)

    # test extract feat
    feats = recognizer.extract_feat(imgs)
    assert len(feats) == 4

    assert feats[0].shape == torch.Size([1, 128, 32, 128])
    assert feats[1].shape == torch.Size([1, 256, 16, 64])
    assert feats[2].shape == torch.Size([1, 512, 8, 32])
    assert feats[3].shape == torch.Size([1, 512, 4, 16])

    attn_tgt = np.zeros((64, 256), dtype=np.float32)
    segm_tgt = np.zeros((64, 256), dtype=np.float32)
    mask = np.zeros((64, 256), dtype=np.float32)
    gt_kernels = BitmapMasks([attn_tgt, segm_tgt, mask], 64, 256)

    # test forward train
    img_metas = [{
        'text': 'hello',
        'resize_shape': (64, 256, 3),
        'valid_ratio': 1.0
    }]
    losses = recognizer.forward_train(imgs, img_metas, gt_kernels=[gt_kernels])
    assert isinstance(losses, dict)

    # test simple test
    results = recognizer.simple_test(imgs, img_metas)
    assert isinstance(results, list)
    assert isinstance(results[0], dict)
    assert 'text' in results[0]
    assert 'score' in results[0]

    # test aug_test
    aug_results = recognizer.aug_test([imgs, imgs], [img_metas, img_metas])
    assert isinstance(aug_results, list)
    assert isinstance(aug_results[0], dict)
    assert 'text' in aug_results[0]
    assert 'score' in aug_results[0]

    tmp_dir.cleanup()
