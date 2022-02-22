# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings

import mmcv
import numpy as np
import torch
from mmdet.datasets import replace_ImageToTensor

from mmocr.utils import is_2dlist, is_type_list


def update_pipeline(cfg, idx=None):
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


def update_pipeline_recog(cfg, idx=None):
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


def tensor2grayimgs(tensor, mean=(127, ), std=(127, ), **kwargs):
    """Convert tensor to 1-channel gray images.

    Args:
        tensor (torch.Tensor): Tensor that contains multiple images, shape (
            N, C, H, W).
        mean (tuple[float], optional): Mean of images. Defaults to (127).
        std (tuple[float], optional): Standard deviation of images.
            Defaults to (127).

    Returns:
        list[np.ndarray]: A list that contains multiple images.
    """

    assert torch.is_tensor(tensor) and tensor.ndim == 4
    assert tensor.size(1) == len(mean) == len(std) == 1

    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(img, mean, std, to_bgr=False).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs
