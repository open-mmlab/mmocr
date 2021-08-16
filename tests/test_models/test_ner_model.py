# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import tempfile

import pytest
import torch

from mmocr.models import build_detector


def _create_dummy_vocab_file(vocab_file):
    with open(vocab_file, 'w') as fw:
        for char in list(map(chr, range(ord('a'), ord('z') + 1))):
            fw.write(char + '\n')


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
    'cfg_file', ['configs/ner/bert_softmax/bert_softmax_cluener_18e.py'])
def test_bert_softmax(cfg_file):
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

    # create dummy data
    tmp_dir = tempfile.TemporaryDirectory()
    vocab_file = osp.join(tmp_dir.name, 'fake_vocab.txt')
    _create_dummy_vocab_file(vocab_file)

    model = _get_detector_cfg(cfg_file)
    model['label_convertor']['vocab_file'] = vocab_file

    detector = build_detector(model)
    losses = detector.forward(img, img_metas)
    assert isinstance(losses, dict)

    model['loss']['type'] = 'MaskedFocalLoss'
    detector = build_detector(model)
    losses = detector.forward(img, img_metas)
    assert isinstance(losses, dict)

    tmp_dir.cleanup()

    # Test forward test
    with torch.no_grad():
        batch_results = []
        result = detector.forward(None, img_metas, return_loss=False)
        batch_results.append(result)
