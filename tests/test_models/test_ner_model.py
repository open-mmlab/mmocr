import copy
from os.path import dirname, exists, join

import pytest
import torch


def _get_config_directory():
    """Find the predefined detector config directory."""
    try:
        # Assume we are running in the source mmocr repo
        repo_dpath = dirname(dirname(dirname(__file__)))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmocr
        repo_dpath = dirname(dirname(mmocr.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmcv import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod


def _get_detector_cfg(fname):
    """Grab configs necessary to create a detector.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    return model


@pytest.mark.parametrize('cfg_file', ['ner/ner_task.py'])
def test_encoder_decoder_pipeline(cfg_file):
    model = _get_detector_cfg(cfg_file)
    model['pretrained'] = None

    from mmocr.models import build_detector
    detector = build_detector(model)
    texts = ['中'] * 47
    img = [31] * 47
    labels = [31] * 128
    input_ids = [0] * 128
    attention_mask = [0] * 128
    token_type_ids = [0] * 128

    # test forward train
    img_metas = [{
        'texts': texts,
        'labels': labels,
        'img': img,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids
    }]
    # Test forward train
    losses = detector.forward(img, img_metas)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        batch_results = []
        result = detector.forward([img], img_metas, return_loss=False)
        batch_results.append(result)
