import pytest

from mmocr.core import prepare_det_model, prepare_recog_model


def test_prepare_model():
    det_config, det_ckpt = prepare_det_model(model_type='panet')
    recog_config, recog_ckpt = prepare_recog_model(model_type='sar')

    with pytest.raises(Exception):
        prepare_det_model(model_type='det')
    with pytest.raises(Exception):
        prepare_recog_model(model_type='recog')
