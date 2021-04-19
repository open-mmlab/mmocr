import os
import shutil
import urllib

import pytest
from mmcv.image import imread

from mmdet.apis import init_detector
from mmocr.apis.inference import model_inference


@pytest.fixture
def project_dir():
    return os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def sample_img_path(project_dir):
    return os.path.join(project_dir, '../demo/demo_text_recog.jpg')


@pytest.fixture
def sample_det_img_path(project_dir):
    return os.path.join(project_dir, '../demo/demo_text_det.jpg')


@pytest.fixture
def sarnet_model(project_dir):
    print(project_dir)
    config_file = os.path.join(
        project_dir,
        '../configs/textrecog/sar/sar_r31_parallel_decoder_academic.py')
    checkpoint_file = os.path.join(
        project_dir,
        '../checkpoints/sar_r31_parallel_decoder_academic-dba3a4a3.pth')

    if not os.path.exists(checkpoint_file):
        url = ('https://download.openmmlab.com/mmocr'
               '/textrecog/sar/'
               'sar_r31_parallel_decoder_academic-dba3a4a3.pth')
        print(f'Downloading {url} ...')
        local_filename, _ = urllib.request.urlretrieve(url)
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        shutil.move(local_filename, checkpoint_file)
        print(f'Saved as {checkpoint_file}')
    else:
        print(f'Using existing checkpoint {checkpoint_file}')

    device = 'cpu'
    model = init_detector(
        config_file, checkpoint=checkpoint_file, device=device)
    if model.cfg.data.test['type'] == 'ConcatDataset':
        model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][
            0].pipeline

    return model


@pytest.fixture
def psenet_model(project_dir):
    config_file = os.path.join(
        project_dir,
        '../configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2017.py')

    device = 'cpu'
    model = init_detector(config_file, checkpoint=None, device=device)
    if model.cfg.data.test['type'] == 'ConcatDataset':
        model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][
            0].pipeline

    return model


def test_model_inference_image_path(sample_img_path, sarnet_model):

    with pytest.raises(AssertionError):
        model_inference(sarnet_model, 1)

    model_inference(sarnet_model, sample_img_path)


def test_model_inference_image_path_det(sample_det_img_path, psenet_model):
    model_inference(psenet_model, sample_det_img_path)


def test_model_inference_numpy_ndarray(sample_img_path, sarnet_model):
    img = imread(sample_img_path)
    model_inference(sarnet_model, img)


def test_model_inference_numpy_ndarray_det(sample_det_img_path, psenet_model):
    det_img = imread(sample_det_img_path)
    model_inference(psenet_model, det_img)
