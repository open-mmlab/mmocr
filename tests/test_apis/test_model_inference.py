import os

import pytest
from mmcv.image import imread

from mmdet.apis import init_detector
from mmocr.apis.inference import model_inference


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
    device = 'cpu'
    model = init_detector(config_file, checkpoint=None, device=device)
    if model.cfg.data.test['type'] == 'ConcatDataset':
        model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][
            0].pipeline
    with pytest.raises(AssertionError):
        model_inference(model, 1)

    sample_img_path = os.path.join(tmp_dir, '../demo/demo_text_det.jpg')
    model_inference(model, sample_img_path)

    # numpy inference
    img = imread(sample_img_path)

    model_inference(model, img)
