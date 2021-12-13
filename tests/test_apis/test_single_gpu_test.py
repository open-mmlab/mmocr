# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import tempfile

import pytest
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel

from mmocr.apis.test import single_gpu_test
from mmocr.datasets import build_dataloader, build_dataset
from mmocr.models import build_detector
from mmocr.utils import check_argument, revert_sync_batchnorm


def build_model(cfg):
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    model = revert_sync_batchnorm(model)
    model = MMDataParallel(model)

    return model


def gene_sample_dataloader(cfg, curr_dir, img_prefix='', ann_file=''):
    img_prefix = osp.join(curr_dir, img_prefix)
    ann_file = osp.join(curr_dir, ann_file)
    test = copy.deepcopy(cfg.data.test.datasets[0])
    test.img_prefix = img_prefix
    test.ann_file = ann_file

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
    data_loader = gene_sample_dataloader(cfg, curr_dir, img_prefix, ann_file)

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
    data_loader = gene_sample_dataloader(cfg, curr_dir, img_prefix, ann_file)

    with tempfile.TemporaryDirectory() as tmpdirname:
        out_dir = osp.join(tmpdirname, 'tmp')
        results = single_gpu_test(model, data_loader, out_dir=out_dir)
        assert check_argument.is_type_list(results, dict)
