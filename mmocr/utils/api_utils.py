# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings

from mmocr.utils.check_argument import is_2dlist, is_type_list

# TODO Checking if the utils will be used after refactoring the inference.py
# If kept, add Tests.


def update_pipeline_recog(cfg, idx=None):
    warning_msg = 'Remove "MultiRotateAugOCR" to support batch ' + \
        'inference since samples_per_gpu > 1.'
    if idx is None:
        if cfg.get('pipeline',
                   None) and cfg.pipeline[1].type == 'MultiRotateAugOCR':
            warnings.warn(warning_msg)
            cfg.pipeline = [cfg.pipeline[0], *cfg.pipeline[1].transforms]
    else:
        if cfg[idx][1].type == 'MultiRotateAugOCR':
            warnings.warn(warning_msg)
            cfg[idx] = [cfg[idx][0], *cfg[idx][1].transforms]


def disable_text_recog_aug_test(cfg, set_types=None):
    """Remove aug_test from test pipeline for text recognition.

    Args:
        cfg (mmcv.Config): Input config.
        set_types (list[str]): Type of dataset source. Should be
            None or sublist of ['test', 'val'].
    """
    assert set_types is None or isinstance(set_types, list)
    if set_types is None:
        set_types = ['val', 'test']

    cfg = copy.deepcopy(cfg)
    warnings.simplefilter('once')
    for set_type in set_types:
        assert set_type in ['val', 'test']
        dataset_type = cfg.data[set_type].type
        if dataset_type not in [
                'ConcatDataset', 'UniformConcatDataset', 'OCRDataset',
                'OCRSegDataset'
        ]:
            continue

        uniform_pipeline = cfg.data[set_type].get('pipeline', None)
        if is_type_list(uniform_pipeline, dict):
            update_pipeline_recog(cfg.data[set_type])
        elif is_2dlist(uniform_pipeline):
            for idx, _ in enumerate(uniform_pipeline):
                update_pipeline_recog(cfg.data[set_type].pipeline, idx)

        for dataset in cfg.data[set_type].get('datasets', []):
            if isinstance(dataset, list):
                for each_dataset in dataset:
                    update_pipeline_recog(each_dataset)
            else:
                update_pipeline_recog(dataset)

    return cfg
