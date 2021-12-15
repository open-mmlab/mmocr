# Copyright (c) OpenMMLab. All rights reserved.
import os

import pytest
from mmcv.image import imread

from mmocr.apis.inference import init_detector, model_inference
from mmocr.datasets import build_dataset  # noqa: F401
from mmocr.models import build_detector  # noqa: F401
from mmocr.utils import revert_sync_batchnorm


def build_model(config_file):
    device = 'cpu'
    model = init_detector(config_file, checkpoint=None, device=device)
    model = revert_sync_batchnorm(model)

    if model.cfg.data.test.get('pipeline', None) is None:
        model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][
            0].pipeline

    return model


@pytest.mark.parametrize('cfg_file', [
    '../configs/textrecog/sar/sar_r31_parallel_decoder_academic.py',
    '../configs/textrecog/abinet/abinet_academic.py',
    '../configs/textrecog/crnn/crnn_academic_dataset.py',
    '../configs/textrecog/seg/seg_r31_1by16_fpnocr_academic.py',
    '../configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2017.py'
])
def test_model_inference(cfg_file):
    tmp_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    config_file = os.path.join(tmp_dir, cfg_file)
    model = build_model(config_file)
    with pytest.raises(AssertionError):
        model_inference(model, 1)

    sample_img_path = os.path.join(tmp_dir, '../demo/demo_text_det.jpg')
    model_inference(model, sample_img_path)

    # numpy inference
    img = imread(sample_img_path)

    model_inference(model, img)


@pytest.mark.parametrize(
    'cfg_file',
    ['../configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2017.py'])
def test_model_batch_inference_det(cfg_file):
    tmp_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    config_file = os.path.join(tmp_dir, cfg_file)
    model = build_model(config_file)

    sample_img_path = os.path.join(tmp_dir, '../demo/demo_text_det.jpg')
    results = model_inference(model, [sample_img_path], batch_mode=True)

    assert len(results) == 1

    # numpy inference
    img = imread(sample_img_path)
    results = model_inference(model, [img], batch_mode=True)

    assert len(results) == 1


@pytest.mark.parametrize('cfg_file', [
    '../configs/textrecog/sar/sar_r31_parallel_decoder_academic.py',
])
def test_model_batch_inference_raises_exception_error_aug_test_recog(cfg_file):
    tmp_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    config_file = os.path.join(tmp_dir, cfg_file)
    model = build_model(config_file)

    with pytest.raises(
            Exception,
            match='aug test does not support inference with batch size'):
        sample_img_path = os.path.join(tmp_dir, '../demo/demo_text_det.jpg')
        model_inference(model, [sample_img_path, sample_img_path])

    with pytest.raises(
            Exception,
            match='aug test does not support inference with batch size'):
        img = imread(sample_img_path)
        model_inference(model, [img, img])


@pytest.mark.parametrize('cfg_file', [
    '../configs/textrecog/sar/sar_r31_parallel_decoder_academic.py',
])
def test_model_batch_inference_recog(cfg_file):
    tmp_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    config_file = os.path.join(tmp_dir, cfg_file)
    model = build_model(config_file)

    sample_img_path = os.path.join(tmp_dir, '../demo/demo_text_recog.jpg')
    results = model_inference(
        model, [sample_img_path, sample_img_path], batch_mode=True)

    assert len(results) == 2

    # numpy inference
    img = imread(sample_img_path)
    results = model_inference(model, [img, img], batch_mode=True)

    assert len(results) == 2


@pytest.mark.parametrize(
    'cfg_file',
    ['../configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2017.py'])
def test_model_batch_inference_empty_detection(cfg_file):
    tmp_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    config_file = os.path.join(tmp_dir, cfg_file)
    model = build_model(config_file)

    empty_detection = []

    with pytest.raises(
            Exception,
            match='empty imgs provided, please check and try again'):

        model_inference(model, empty_detection, batch_mode=True)
