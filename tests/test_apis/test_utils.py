# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os

import pytest
from mmcv import Config

from mmocr.apis.utils import (disable_text_recog_aug_test,
                              replace_image_to_tensor)


@pytest.mark.parametrize('cfg_file', [
    '../configs/textrecog/sar/sar_r31_parallel_decoder_academic.py',
])
def test_disable_text_recog_aug_test(cfg_file):
    tmp_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    config_file = os.path.join(tmp_dir, cfg_file)

    cfg = Config.fromfile(config_file)
    test = cfg.data.test.datasets[0]

    # cfg.data.test.type is 'OCRDataset'
    cfg1 = copy.deepcopy(cfg)
    test1 = copy.deepcopy(test)
    test1.pipeline = cfg1.data.test.pipeline
    cfg1.data.test = test1
    cfg1 = disable_text_recog_aug_test(cfg1, set_types=['test'])
    assert cfg1.data.test.pipeline[1].rotate_degrees == [0]

    # cfg.data.test.type is 'UniformConcatDataset'
    # and cfg.data.test.pipeline is list[dict]
    cfg2 = copy.deepcopy(cfg)
    test2 = copy.deepcopy(test)
    test2.pipeline = cfg2.data.test.pipeline
    cfg2.data.test.datasets = [test2]
    cfg2 = disable_text_recog_aug_test(cfg2, set_types=['test'])
    assert cfg2.data.test.pipeline[1].rotate_degrees == [0]
    assert cfg2.data.test.datasets[0].pipeline[1].rotate_degrees == [0]

    # cfg.data.test.type is 'ConcatDataset'
    cfg3 = copy.deepcopy(cfg)
    test3 = copy.deepcopy(test)
    test3.pipeline = cfg3.data.test.pipeline
    cfg3.data.test = Config(dict(type='ConcatDataset', datasets=[test3]))
    cfg3 = disable_text_recog_aug_test(cfg3, set_types=['test'])
    assert cfg3.data.test.datasets[0].pipeline[1].rotate_degrees == [0]

    # cfg.data.test.type is 'UniformConcatDataset'
    # and cfg.data.test.pipeline is list[list[dict]]
    cfg4 = copy.deepcopy(cfg)
    test4 = copy.deepcopy(test)
    test4.pipeline = cfg4.data.test.pipeline
    cfg4.data.test.datasets = [[test4], [test]]
    cfg4.data.test.pipeline = [
        cfg4.data.test.pipeline, cfg4.data.test.pipeline
    ]
    cfg4 = disable_text_recog_aug_test(cfg4, set_types=['test'])
    assert cfg4.data.test.datasets[0][0].pipeline[1].rotate_degrees == \
        [0]

    # cfg.data.test.type is 'UniformConcatDataset'
    # and cfg.data.test.pipeline is None
    cfg5 = copy.deepcopy(cfg)
    test5 = copy.deepcopy(test)
    test5.pipeline = copy.deepcopy(cfg5.data.test.pipeline)
    cfg5.data.test.datasets = [test5]
    cfg5.data.test.pipeline = None
    cfg5 = disable_text_recog_aug_test(cfg5, set_types=['test'])
    assert cfg5.data.test.datasets[0].pipeline[1].rotate_degrees == [0]


@pytest.mark.parametrize('cfg_file', [
    '../configs/textdet/psenet/psenet_r50_fpnf_600e_ctw1500.py',
])
def test_replace_image_to_tensor(cfg_file):
    tmp_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    config_file = os.path.join(tmp_dir, cfg_file)

    cfg = Config.fromfile(config_file)
    test = cfg.data.test.datasets[0]

    # cfg.data.test.pipeline is list[dict]
    # and cfg.data.test.datasets is list[dict]
    cfg1 = copy.deepcopy(cfg)
    test1 = copy.deepcopy(test)
    test1.pipeline = copy.deepcopy(cfg.data.test.pipeline)
    cfg1.data.test.datasets = [test1]
    cfg1 = replace_image_to_tensor(cfg1, set_types=['test'])
    assert cfg1.data.test.pipeline[1]['transforms'][3][
        'type'] == 'DefaultFormatBundle'
    assert cfg1.data.test.datasets[0].pipeline[1]['transforms'][3][
        'type'] == 'DefaultFormatBundle'

    # cfg.data.test.pipeline is list[list[dict]]
    # and cfg.data.test.datasets is list[list[dict]]
    cfg2 = copy.deepcopy(cfg)
    test2 = copy.deepcopy(test)
    test2.pipeline = copy.deepcopy(cfg.data.test.pipeline)
    cfg2.data.test.datasets = [[test2], [test2]]
    cfg2.data.test.pipeline = [
        cfg2.data.test.pipeline, cfg2.data.test.pipeline
    ]
    cfg2 = replace_image_to_tensor(cfg2, set_types=['test'])
    assert cfg2.data.test.pipeline[0][1]['transforms'][3][
        'type'] == 'DefaultFormatBundle'
    assert cfg2.data.test.datasets[0][0].pipeline[1]['transforms'][3][
        'type'] == 'DefaultFormatBundle'
