# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmocr.models.textrecog.preprocessor import (BasePreprocessor,
                                                 TPSPreprocessor)


def test_tps_preprocessor():
    with pytest.raises(AssertionError):
        TPSPreprocessor(num_fiducial=-1)
    with pytest.raises(AssertionError):
        TPSPreprocessor(img_size=32)
    with pytest.raises(AssertionError):
        TPSPreprocessor(rectified_img_size=100)
    with pytest.raises(AssertionError):
        TPSPreprocessor(num_img_channel='bgr')

    tps_preprocessor = TPSPreprocessor(
        num_fiducial=20,
        img_size=(32, 100),
        rectified_img_size=(32, 100),
        num_img_channel=1)
    tps_preprocessor.init_weights()
    tps_preprocessor.train()

    batch_img = torch.randn(1, 1, 32, 100)
    processed = tps_preprocessor(batch_img)
    assert processed.shape == torch.Size([1, 1, 32, 100])


def test_base_preprocessor():
    preprocessor = BasePreprocessor()
    preprocessor.init_weights()
    preprocessor.train()

    batch_img = torch.randn(1, 1, 32, 100)
    processed = preprocessor(batch_img)
    assert processed.shape == torch.Size([1, 1, 32, 100])
