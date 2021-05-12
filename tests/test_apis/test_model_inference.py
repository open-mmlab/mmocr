import os

import pytest
from mmcv.image import imread

from mmdet.apis import init_detector
from mmocr.apis.inference import model_inference
from mmocr.datasets import build_dataset  # noqa: F401
from mmocr.models import build_detector  # noqa: F401


def build_model(config_file):
    device = 'cpu'
    model = init_detector(config_file, checkpoint=None, device=device)

    if model.cfg.data.test['type'] == 'ConcatDataset':
        model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][
            0].pipeline

    return model


def disable_aug_test(model):
    model.cfg.data.test.pipeline = [
        model.cfg.data.test.pipeline[0],
        *model.cfg.data.test.pipeline[1].transforms
    ]

    return model


@pytest.mark.parametrize('cfg_file', [
    '../configs/textrecog/sar/sar_r31_parallel_decoder_academic.py',
    '../configs/textrecog/crnn/crnn_academic_dataset.py',
    '../configs/textrecog/nrtr/nrtr_r31_1by16_1by8_academic.py',
    '../configs/textrecog/robust_scanner/robustscanner_r31_academic.py',
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


@pytest.mark.parametrize('cfg_file', [
    '../configs/textrecog/crnn/crnn_academic_dataset.py',
    '../configs/textrecog/seg/seg_r31_1by16_fpnocr_academic.py'
])
def test_model_batch_inference(cfg_file):
    tmp_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    config_file = os.path.join(tmp_dir, cfg_file)
    model = build_model(config_file)

    sample_img_path = os.path.join(tmp_dir, '../demo/demo_text_det.jpg')
    results = model_inference(model, [sample_img_path, sample_img_path])

    assert len(results) == 2

    # numpy inference
    img = imread(sample_img_path)
    results = model_inference(model, [img, img])

    assert len(results) == 2


@pytest.mark.parametrize('cfg_file', [
    '../configs/textrecog/sar/sar_r31_parallel_decoder_academic.py',
    '../configs/textrecog/nrtr/nrtr_r31_1by16_1by8_academic.py',
    '../configs/textrecog/robust_scanner/robustscanner_r31_academic.py',
])
def test_model_batch_inference_raises_assertion_error_if_unsupported(cfg_file):
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
    '../configs/textrecog/nrtr/nrtr_r31_1by16_1by8_academic.py',
    '../configs/textrecog/robust_scanner/robustscanner_r31_academic.py',
])
def test_model_batch_inference_recog(cfg_file):
    tmp_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    config_file = os.path.join(tmp_dir, cfg_file)
    model = build_model(config_file)
    model = disable_aug_test(model)

    sample_img_path = os.path.join(tmp_dir, '../demo/demo_text_det.jpg')
    results = model_inference(model, [sample_img_path, sample_img_path])

    assert len(results) == 2

    # numpy inference
    img = imread(sample_img_path)
    results = model_inference(model, [img, img])

    assert len(results) == 2
