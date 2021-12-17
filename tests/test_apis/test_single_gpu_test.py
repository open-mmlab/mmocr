# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import os
import os.path as osp
import tempfile

import mmcv
import numpy as np
import pytest
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel

from mmocr.apis.test import single_gpu_test
from mmocr.datasets import build_dataloader, build_dataset
from mmocr.models import build_detector
from mmocr.utils import check_argument, list_to_file, revert_sync_batchnorm


def build_model(cfg):
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    model = revert_sync_batchnorm(model)
    model = MMDataParallel(model)

    return model


def generate_sample_dataloader(cfg, curr_dir, img_prefix='', ann_file=''):
    must_keys = ['img_norm_cfg', 'ori_filename', 'img_shape', 'ori_shape']
    test_pipeline = cfg.data.test.pipeline
    for key in must_keys:
        if test_pipeline[1].type == 'MultiRotateAugOCR':
            collect_pipeline = test_pipeline[1]['transforms'][-1]
        else:
            collect_pipeline = test_pipeline[-1]
        if 'meta_keys' not in collect_pipeline:
            continue
        collect_pipeline['meta_keys'].append(key)

    img_prefix = osp.join(curr_dir, img_prefix)
    ann_file = osp.join(curr_dir, ann_file)
    test = copy.deepcopy(cfg.data.test.datasets[0])
    test.img_prefix = img_prefix
    test.ann_file = ann_file
    cfg.data.workers_per_gpu = 1
    cfg.data.test.datasets = [test]
    dataset = build_dataset(cfg.data.test)

    loader_cfg = {
        **dict((k, cfg.data[k]) for k in [
                   'workers_per_gpu', 'samples_per_gpu'
               ] if k in cfg.data)
    }
    test_loader_cfg = {
        **loader_cfg,
        **dict(shuffle=False, drop_last=False),
        **cfg.data.get('test_dataloader', {})
    }

    data_loader = build_dataloader(dataset, **test_loader_cfg)

    return data_loader


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
@pytest.mark.parametrize('cfg_file', [
    '../configs/textrecog/sar/sar_r31_parallel_decoder_academic.py',
    '../configs/textrecog/crnn/crnn_academic_dataset.py',
    '../configs/textrecog/seg/seg_r31_1by16_fpnocr_academic.py'
])
def test_single_gpu_test_recog(cfg_file):
    curr_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    config_file = os.path.join(curr_dir, cfg_file)
    cfg = Config.fromfile(config_file)

    model = build_model(cfg)
    img_prefix = 'data/ocr_toy_dataset/imgs'
    ann_file = 'data/ocr_toy_dataset/label.txt'
    data_loader = generate_sample_dataloader(cfg, curr_dir, img_prefix,
                                             ann_file)

    with tempfile.TemporaryDirectory() as tmpdirname:
        out_dir = osp.join(tmpdirname, 'tmp')
        results = single_gpu_test(model, data_loader, out_dir=out_dir)
        assert check_argument.is_type_list(results, dict)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
@pytest.mark.parametrize(
    'cfg_file',
    ['../configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2017.py'])
def test_single_gpu_test_det(cfg_file):
    curr_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    config_file = os.path.join(curr_dir, cfg_file)
    cfg = Config.fromfile(config_file)

    model = build_model(cfg)
    img_prefix = 'data/toy_dataset/imgs'
    ann_file = 'data/toy_dataset/instances_test.json'
    data_loader = generate_sample_dataloader(cfg, curr_dir, img_prefix,
                                             ann_file)

    with tempfile.TemporaryDirectory() as tmpdirname:
        out_dir = osp.join(tmpdirname, 'tmp')
        results = single_gpu_test(model, data_loader, out_dir=out_dir)
        assert check_argument.is_type_list(results, dict)


def gene_sdmgr_model_dataloader(cfg, dirname, curr_dir, empty_img=False):
    json_obj = {
        'file_name':
        '1.jpg',
        'height':
        348,
        'width':
        348,
        'annotations': [{
            'box': [114.0, 19.0, 230.0, 19.0, 230.0, 1.0, 114.0, 1.0],
            'text':
            'CHOEUN',
            'label':
            1
        }]
    }
    ann_file = osp.join(dirname, 'test.txt')
    list_to_file(ann_file, [json.dumps(json_obj, ensure_ascii=False)])

    if not empty_img:
        img = np.ones((348, 348, 3), dtype=np.uint8)
        img_file = osp.join(dirname, '1.jpg')
        mmcv.imwrite(img, img_file)

    test = copy.deepcopy(cfg.data.test)
    test.ann_file = ann_file
    test.img_prefix = dirname
    test.dict_file = osp.join(curr_dir, 'data/kie_toy_dataset/dict.txt')
    cfg.data.workers_per_gpu = 1
    cfg.data.test = test
    cfg.model.class_list = osp.join(curr_dir,
                                    'data/kie_toy_dataset/class_list.txt')

    dataset = build_dataset(cfg.data.test)

    loader_cfg = {
        **dict((k, cfg.data[k]) for k in [
                   'workers_per_gpu', 'samples_per_gpu'
               ] if k in cfg.data)
    }
    test_loader_cfg = {
        **loader_cfg,
        **dict(shuffle=False, drop_last=False),
        **cfg.data.get('test_dataloader', {})
    }

    data_loader = build_dataloader(dataset, **test_loader_cfg)
    model = build_model(cfg)

    return model, data_loader


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
@pytest.mark.parametrize(
    'cfg_file', ['../configs/kie/sdmgr/sdmgr_unet16_60e_wildreceipt.py'])
def test_single_gpu_test_kie(cfg_file):
    curr_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    config_file = os.path.join(curr_dir, cfg_file)
    cfg = Config.fromfile(config_file)

    with tempfile.TemporaryDirectory() as tmpdirname:
        out_dir = osp.join(tmpdirname, 'tmp')
        model, data_loader = gene_sdmgr_model_dataloader(
            cfg, out_dir, curr_dir)
        results = single_gpu_test(
            model, data_loader, out_dir=out_dir, is_kie=True)
        assert check_argument.is_type_list(results, dict)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
@pytest.mark.parametrize(
    'cfg_file', ['../configs/kie/sdmgr/sdmgr_novisual_60e_wildreceipt.py'])
def test_single_gpu_test_kie_novisual(cfg_file):
    curr_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    config_file = os.path.join(curr_dir, cfg_file)
    cfg = Config.fromfile(config_file)
    meta_keys = list(cfg.data.test.pipeline[-1]['meta_keys'])
    must_keys = ['img_norm_cfg', 'ori_filename', 'img_shape']
    for key in must_keys:
        meta_keys.append(key)

    cfg.data.test.pipeline[-1]['meta_keys'] = tuple(meta_keys)

    with tempfile.TemporaryDirectory() as tmpdirname:
        out_dir = osp.join(tmpdirname, 'tmp')
        model, data_loader = gene_sdmgr_model_dataloader(
            cfg, out_dir, curr_dir, empty_img=True)
        results = single_gpu_test(
            model, data_loader, out_dir=out_dir, is_kie=True)
        assert check_argument.is_type_list(results, dict)

        model, data_loader = gene_sdmgr_model_dataloader(
            cfg, out_dir, curr_dir)
        results = single_gpu_test(
            model, data_loader, out_dir=out_dir, is_kie=True)
        assert check_argument.is_type_list(results, dict)
