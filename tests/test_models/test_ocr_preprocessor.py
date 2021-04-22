import torch

from mmocr.models.textrecog.preprocessor import (BasePreprocessor,
                                                 TPSPreprocessor)


def test_tps_preprocessor():
    tps_preprocessor = TPSPreprocessor(
        num_fiducial=20,
        img_size=(32, 100),
        img_rectified_size=(32, 100),
        img_channel_num=1)
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
