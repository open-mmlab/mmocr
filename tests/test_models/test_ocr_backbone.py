# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmocr.models.textrecog.backbones import (ResNet31OCR, ResNetABI,
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
