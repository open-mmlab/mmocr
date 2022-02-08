# Copyright (c) OpenMMLab. All rights reserved.
import copy
from os.path import dirname, exists, join

import pytest
from mmcv import Config

from mmocr.utils import unify_recog_pipeline


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


@pytest.mark.parametrize('cfg_file',
                         ['textrecog/crnn/crnn_academic_dataset.py'])
def test_unify_recog_pipeline(cfg_file):
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, cfg_file)
    default_cfg = Config.fromfile(config_fpath)

    cfg = copy.deepcopy(default_cfg)
    unified_cfg = unify_recog_pipeline(cfg)
    assert unified_cfg.data.test.pipeline[1]['type'] == 'MultiRotateAugOCR'

    test_pipeline = copy.deepcopy(cfg.data.test.pipeline)
    test_pipeline = [test_pipeline[0], *test_pipeline[1].transforms]
    new_pipeline = []
    for pipeline in test_pipeline:
        if pipeline.type in ['ResizeOCR', 'Collect']:
            new_pipeline.append(pipeline)
        elif pipeline.type == 'Normalize':
            new_pipeline.append(dict(type='ToTensorOCR'))
            new_pipeline.append(
                dict(type='NormalizeOCR', mean=[127], std=[127]))

    cfg.data.test.pipeline = new_pipeline
    unified_cfg = unify_recog_pipeline(cfg)
    assert unified_cfg.data.test.pipeline[1]['type'] == 'MultiRotateAugOCR'

    cfg = copy.deepcopy(default_cfg)
    cfg.data.test.pipeline = [new_pipeline]
    cfg.data.test.datasets[0].pipeline = new_pipeline
    unified_cfg = unify_recog_pipeline(cfg)
    assert unified_cfg.data.test.pipeline[0][1]['type'] == 'MultiRotateAugOCR'
    assert unified_cfg.data.test.datasets[0].pipeline[1][
        'type'] == 'MultiRotateAugOCR'
