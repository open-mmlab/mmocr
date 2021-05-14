import copy

import pytest
import torch

from mmocr.models import build_detector
from mmocr.models.ner.utils.activations import gelu, gelu_new, swish


def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmcv import Config
    config_mod = Config.fromfile(fname)
    return config_mod


def _get_detector_cfg(fname):
    """Grab configs necessary to create a detector.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    return model


@pytest.mark.parametrize(
    'cfg_file', ['configs/ner/bert_softmax/bert_softmax_toy_dataset.py'])
def test_encoder_decoder_pipeline(cfg_file):
    # prepare data
    texts = ['中'] * 47
    img = [31] * 47
    labels = [31] * 128
    input_ids = [0] * 128
    attention_mask = [0] * 128
    token_type_ids = [0] * 128
    img_metas = {
        'texts': texts,
        'labels': torch.tensor(labels).unsqueeze(0),
        'img': img,
        'input_ids': torch.tensor(input_ids).unsqueeze(0),
        'attention_masks': torch.tensor(attention_mask).unsqueeze(0),
        'token_type_ids': torch.tensor(token_type_ids).unsqueeze(0)
    }

    model = _get_detector_cfg(cfg_file)
    model['pretrained'] = None

    detector = build_detector(model)
    losses = detector.forward(img, img_metas)
    assert isinstance(losses, dict)

    model['loss']['type'] = 'MaskedFocalLoss'
    detector = build_detector(model)
    losses = detector.forward(img, img_metas)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        batch_results = []
        result = detector.forward(None, img_metas, return_loss=False)
        batch_results.append(result)

    # Test activations
    gelu(torch.tensor(0.5))
    gelu_new(torch.tensor(0.5))
    swish(torch.tensor(0.5))
