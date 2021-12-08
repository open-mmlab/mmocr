import warnings

from mmdet.datasets import replace_ImageToTensor

from mmocr.utils import is_2dlist


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
    for set_type in set_types:
        assert set_type in ['val', 'test']
        uniform_pipeline = cfg.data[set_type].get('pipeline', None)
        if uniform_pipeline is not None:
            if is_2dlist(uniform_pipeline):
                for idx, _ in enumerate(uniform_pipeline):
                    update_pipeline(cfg.data[set_type], idx)
            else:
                update_pipeline(cfg.data[set_type])

        dataset_type = cfg.data[set_type].type
        if dataset_type in ['ConcatDataset', 'UniformConcatDataset']:
            for dataset in cfg.data[set_type].datasets:
                if isinstance(dataset, list):
                    for each_dataset in dataset:
                        update_pipeline(each_dataset)
                else:
                    update_pipeline(dataset)


def remove_aug_test(cfg, idx=None):
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
    warnings.simplefilter('once')
    for set_type in set_types:
        dataset_type = cfg.data[set_type].type
        if dataset_type in ['OCRDataset', 'OCRSegDataset']:
            remove_aug_test(cfg.data[set_type])
        elif dataset_type in ['ConcatDataset', 'UniformConcatDataset']:
            uniform_pipeline = cfg.data[set_type].get('pipeline', None)
            if uniform_pipeline is not None:
                if is_2dlist(uniform_pipeline):
                    for idx, _ in enumerate(uniform_pipeline):
                        remove_aug_test(cfg.data[set_type].pipeline, idx)
                else:
                    remove_aug_test(cfg.data[set_type])
            for dataset in cfg.data[set_type].datasets:
                if isinstance(dataset, list):
                    for each_dataset in dataset:
                        remove_aug_test(each_dataset)
                else:
                    remove_aug_test(dataset)
