import pytest
import torch

from mmocr.models.textrecog.backbones import ResNet31OCR, VeryDeepVgg


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
