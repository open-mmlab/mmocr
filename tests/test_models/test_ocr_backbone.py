# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmocr.models.textrecog.backbones import (ResNet, ResNet31OCR, ResNetABI,
                                              ShallowCNN, VeryDeepVgg)


def test_resnet31_ocr_backbone():
    """Test resnet backbone."""
    with pytest.raises(AssertionError):
        ResNet31OCR(2.5)

    with pytest.raises(AssertionError):
        ResNet31OCR(3, layers=5)

    with pytest.raises(AssertionError):
        ResNet31OCR(3, channels=5)

    # Test ResNet18 forward
    model = ResNet31OCR()
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 32, 160)
    feat = model(imgs)
    assert feat.shape == torch.Size([1, 512, 4, 40])


def test_vgg_deep_vgg_ocr_backbone():

    model = VeryDeepVgg()
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 32, 160)
    feats = model(imgs)
    assert feats.shape == torch.Size([1, 512, 1, 41])


def test_shallow_cnn_ocr_backbone():

    model = ShallowCNN()
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 1, 32, 100)
    feat = model(imgs)
    assert feat.shape == torch.Size([1, 512, 8, 25])


def test_resnet_abi():
    """Test resnet backbone."""
    with pytest.raises(AssertionError):
        ResNetABI(2.5)

    with pytest.raises(AssertionError):
        ResNetABI(3, arch_settings=5)

    with pytest.raises(AssertionError):
        ResNetABI(3, stem_channels=None)

    with pytest.raises(AssertionError):
        ResNetABI(arch_settings=[3, 4, 6, 6], strides=[1, 2, 1, 2, 1])

    # Test forwarding
    model = ResNetABI()
    model.train()

    imgs = torch.randn(1, 3, 32, 160)
    feat = model(imgs)
    assert feat.shape == torch.Size([1, 512, 8, 40])


def test_resnet():
    """Test all ResNet backbones."""

    resnet45_aster = ResNet(
        in_channels=3,
        stem_channels=[64, 128],
        block_cfgs=dict(type='BasicBlock', use_conv1x1='True'),
        arch_layers=[3, 4, 6, 6, 3],
        arch_channels=[32, 64, 128, 256, 512],
        strides=[(2, 2), (2, 2), (2, 1), (2, 1), (2, 1)])

    resnet45_abi = ResNet(
        in_channels=3,
        stem_channels=32,
        block_cfgs=dict(type='BasicBlock', use_conv1x1='True'),
        arch_layers=[3, 4, 6, 6, 3],
        arch_channels=[32, 64, 128, 256, 512],
        strides=[2, 1, 2, 1, 1])

    resnet_31 = ResNet(
        in_channels=3,
        stem_channels=[64, 128],
        block_cfgs=dict(type='BasicBlock'),
        arch_layers=[1, 2, 5, 3],
        arch_channels=[256, 256, 512, 512],
        strides=[1, 1, 1, 1],
        plugins=[
            dict(
                cfg=dict(type='Maxpool2d', kernel_size=2, stride=(2, 2)),
                stages=(True, True, False, False),
                position='before_stage'),
            dict(
                cfg=dict(type='Maxpool2d', kernel_size=(2, 1), stride=(2, 1)),
                stages=(False, False, True, False),
                position='before_stage'),
            dict(
                cfg=dict(
                    type='ConvModule',
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')),
                stages=(True, True, True, True),
                position='after_stage')
        ])

    resnet31_master = ResNet(
        in_channels=3,
        stem_channels=[64, 128],
        block_cfgs=dict(type='BasicBlock'),
        arch_layers=[1, 2, 5, 3],
        arch_channels=[256, 256, 512, 512],
        strides=[1, 1, 1, 1],
        plugins=[
            dict(
                cfg=dict(type='Maxpool2d', kernel_size=2, stride=(2, 2)),
                stages=(True, True, False, False),
                position='before_stage'),
            dict(
                cfg=dict(type='Maxpool2d', kernel_size=(2, 1), stride=(2, 1)),
                stages=(False, False, True, False),
                position='before_stage'),
            dict(
                cfg=dict(type='GCAModule', ratio=0.0625, n_head=1),
                stages=[True, True, True, True],
                position='after_stage'),
            dict(
                cfg=dict(
                    type='ConvModule',
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')),
                stages=(True, True, True, True),
                position='after_stage')
        ])
    img = torch.rand(1, 3, 32, 100)

    assert resnet45_aster(img).shape == torch.Size([1, 512, 1, 25])
    assert resnet45_abi(img).shape == torch.Size([1, 512, 8, 25])
    assert resnet_31(img).shape == torch.Size([1, 512, 4, 25])
    assert resnet31_master(img).shape == torch.Size([1, 512, 4, 25])
