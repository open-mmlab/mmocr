# Copyright (c) OpenMMLab. All rights reserved.
import copy

from mmcv.utils import Registry, build_from_cfg
from mmdet.datasets import DATASETS as MMDET_DATASETS
from mmdet.datasets import PIPELINES as MMDET_PIPELINES

LOADERS = Registry('loader')
PARSERS = Registry('parser')
DATASETS = Registry('dataset', parent=MMDET_DATASETS)
PIPELINES = Registry('pipeline', parent=MMDET_PIPELINES)


def build_loader(cfg):
    """Build anno file loader."""
    return build_from_cfg(cfg, LOADERS)


def build_parser(cfg):
    """Build anno file parser."""
    return build_from_cfg(cfg, PARSERS)


def _concat_dataset(cfg, default_args=None):
    from mmdet.datasets.dataset_wrappers import ConcatDataset
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    seg_prefixes = cfg.get('seg_prefix', None)
    proposal_files = cfg.get('proposal_file', None)
    separate_eval = cfg.get('separate_eval', True)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        # pop 'separate_eval' since it is not a valid key for common datasets.
        if 'separate_eval' in data_cfg:
            data_cfg.pop('separate_eval')
        data_cfg['ann_file'] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg['img_prefix'] = img_prefixes[i]
        if isinstance(seg_prefixes, (list, tuple)):
            data_cfg['seg_prefix'] = seg_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg['proposal_file'] = proposal_files[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets, separate_eval)


def build_dataset(cfg, default_args=None):
    from mmdet.datasets.dataset_wrappers import (ConcatDataset, RepeatDataset,
                                                 ClassBalancedDataset)
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'mmdet.ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'mmdet.RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'mmdet.ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset
