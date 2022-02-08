# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings

from mmocr.utils import is_2dlist, is_type_list


def unify_recog_pipeline(cfg):
    recog_model_type = [
        'ABINet', 'CRNNNet', 'NRTR', 'RobustScanner', 'SARNet', 'SATRN',
        'SegRecognizer'
    ]
    is_recog = cfg.model.type in recog_model_type
    if not is_recog:
        return cfg

    cfg = copy.deepcopy(cfg)
    for set_type in ['val', 'test']:
        uniform_pipeline = cfg.data[set_type].get('pipeline', None)
        if is_type_list(uniform_pipeline, dict):
            update_pipeline(cfg.data[set_type])
        elif is_2dlist(uniform_pipeline):
            for idx, _ in enumerate(uniform_pipeline):
                update_pipeline(cfg.data[set_type], idx)

        for dataset in cfg.data[set_type].get('datasets', []):
            if isinstance(dataset, list):
                for each_dataset in dataset:
                    update_pipeline(each_dataset)
            else:
                update_pipeline(dataset)

    return cfg


def update_pipeline(cfg, idx=None):
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
        dict(
            type='MultiRotateAugOCR',
            rotate_degrees=rotate_degrees,
            transforms=new_transforms))

    return new_pipelines


def update_transforms(transforms):
    new_transforms = []
    for transform in transforms:
        if transform['type'] in [
                'ResizeOCR', 'Normalize', 'DefaultFormatBundle', 'Collect'
        ]:
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
            new_transforms.append(normalize)
            new_transforms.append(dict(type='DefaultFormatBundle'))

    return new_transforms
