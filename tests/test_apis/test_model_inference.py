import os
import shutil
import urllib

import pytest

from mmdet.apis import init_detector
from mmocr.apis.inference import model_inference


def test_model_inference():

    project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
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

    device = 'cuda:0'
    model = init_detector(
        config_file, checkpoint=checkpoint_file, device=device)
    if model.cfg.data.test['type'] == 'ConcatDataset':
        model.cfg.data.test.pipeline = \
            model.cfg.data.test['datasets'][0].pipeline

    img = os.path.join(project_dir, '../demo/demo_text_recog.jpg')

    with pytest.raises(AssertionError):
        model_inference(model, 1)

    model_inference(model, img)
