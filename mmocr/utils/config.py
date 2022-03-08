# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings

from mmcv import Config
from mmdet.datasets import replace_ImageToTensor

from mmocr.utils import is_2dlist, is_type_list


def _replace_image_to_tensor(cfg, idx=None):
    if idx is None:
        if cfg.pipeline is not None:
            cfg.pipeline = replace_ImageToTensor(cfg.pipeline)
    else:
        cfg.pipeline[idx] = replace_ImageToTensor(cfg.pipeline[idx])


def replace_image_to_tensor(cfg, set_types=None):
    """Replace 'ImageToTensor' to 'DefaultFormatBundle'."""
    assert set_types is None or isinstance(set_types, list)
    if set_types is None:
        set_types = ['val', 'test']

    cfg = copy.deepcopy(cfg)
    for set_type in set_types:
        assert set_type in ['val', 'test']
        uniform_pipeline = cfg.data[set_type].get('pipeline', None)
        if is_type_list(uniform_pipeline, dict):
            _replace_image_to_tensor(cfg.data[set_type])
        elif is_2dlist(uniform_pipeline):
            for idx, _ in enumerate(uniform_pipeline):
                _replace_image_to_tensor(cfg.data[set_type], idx)

        for dataset in cfg.data[set_type].get('datasets', []):
            if isinstance(dataset, list):
                for each_dataset in dataset:
                    _replace_image_to_tensor(each_dataset)
            else:
                _replace_image_to_tensor(dataset)

    return cfg


def _disable_recog_aug_test(cfg, idx=None):
    warning_msg = 'Set "rotate_degrees=[0]" (just one degree) to support ' + \
        'batch inference since samples_per_gpu > 1.'
    if idx is None:
        if cfg.get('pipeline',
                   None) and cfg.pipeline[1].type == 'MultiRotateAugOCR':
            if len(cfg.pipeline[1].get('rotate_degrees', [0])) != 1:
                warnings.warn(warning_msg)
                cfg.pipeline[1]['rotate_degrees'] = [0]
    else:
        if cfg[idx][1].type == 'MultiRotateAugOCR':
            if len(cfg[idx][1].get('rotate_degrees', [0])) != 1:
                warnings.warn(warning_msg)
                cfg[idx][1]['rotate_degrees'] = [0]


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
            _disable_recog_aug_test(cfg.data[set_type])
        elif is_2dlist(uniform_pipeline):
            for idx, _ in enumerate(uniform_pipeline):
                _disable_recog_aug_test(cfg.data[set_type].pipeline, idx)

        for dataset in cfg.data[set_type].get('datasets', []):
            if isinstance(dataset, list):
                for each_dataset in dataset:
                    _disable_recog_aug_test(each_dataset)
            else:
                _disable_recog_aug_test(dataset)

    return cfg


def unify_recog_pipeline(cfg):
    recog_model_type = [
        'CRNNNet', 'SARNet', 'NRTR', 'SegRecognizer', 'RobustScanner', 'SATRN',
        'ABINet'
    ]
    is_recog = cfg.model.type in recog_model_type
    if not is_recog:
        return cfg

    cfg = copy.deepcopy(cfg)
    for set_type in ['val', 'test']:
        uniform_pipeline = cfg.data[set_type].get('pipeline', None)
        if is_type_list(uniform_pipeline, dict):
            _unify_recog_pipeline(cfg.data[set_type])
        elif is_2dlist(uniform_pipeline):
            for idx, _ in enumerate(uniform_pipeline):
                _unify_recog_pipeline(cfg.data[set_type], idx)

        for dataset in cfg.data[set_type].get('datasets', []):
            if isinstance(dataset, list):
                for each_dataset in dataset:
                    _unify_recog_pipeline(each_dataset)
            else:
                _unify_recog_pipeline(dataset)

    return cfg


def _unify_recog_pipeline(cfg, idx=None):
    if idx is None:
        if cfg.pipeline is not None:
            cfg.pipeline = add_aug_test(cfg.pipeline)
    else:
        cfg.pipeline[idx] = add_aug_test(cfg.pipeline[idx])


def add_aug_test(pipelines):
    pipelines = copy.deepcopy(pipelines)
    rotate_degrees = [0]
    if pipelines[1]['type'] == 'MultiRotateAugOCR':
        rotate_degrees = pipelines[1].get('rotate_degrees', [0])
        new_transforms = update_transforms(pipelines[1]['transforms'])
    else:
        warnings.warn(
            '"MultiRotateAugOCR" pipeline must be included '
            'in pipelines. It is recommended to manually add '
            'it in the test data pipeline in your config file. '
            'See https://github.com/open-mmlab/mmocr/pull/740 '
            'for details.', UserWarning)
        new_transforms = update_transforms(pipelines)
    new_pipelines = [pipelines[0]]
    new_pipelines.append(
        Config(
            dict(
                type='MultiRotateAugOCR',
                rotate_degrees=rotate_degrees,
                transforms=new_transforms)))

    return new_pipelines


def update_transforms(transforms):
    new_transforms = []
    for transform in transforms:
        if transform['type'] not in ['ToTensorOCR', 'NormalizeOCR']:
            new_transforms.append(transform)
        if transform['type'] == 'ToTensorOCR':
            warnings.warn(
                '"ToTensorOCR" pipeline is deprecated, please use '
                '"DefaultFormatBundle" for uniform data format. It is '
                'recommended to manually replace it in the test '
                'data pipeline in your config file. '
                'See https://github.com/open-mmlab/mmocr/pull/740 '
                'for details.', UserWarning)
        if transform['type'] == 'NormalizeOCR':
            warnings.warn(
                '"NormalizeOCR" pipeline is deprecated, please use '
                '"Normalize" for unification. It is '
                'recommended to manually replace it in the test '
                'data pipeline in your config file. '
                'See https://github.com/open-mmlab/mmocr/pull/740 '
                'for details.', UserWarning)
            mean = [x if x > 1 else x * 255 for x in transform['mean']]
            std = [x if x > 1 else x * 255 for x in transform['std']]
            normalize = dict(
                type='Normalize', mean=mean, std=std, to_rgb=False)
            new_transforms.append(Config(normalize))
            new_transforms.append(Config(dict(type='DefaultFormatBundle')))

    return new_transforms
