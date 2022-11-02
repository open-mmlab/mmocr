# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from typing import Tuple

import mmengine
from mmengine.config import Config, DictAction

from mmocr.registry import DATASETS, VISUALIZERS
from mmocr.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset.')
    parser.add_argument('config', help='Path to model or dataset config.')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='The interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    # Documentation of the usage of this tool can be found in
    # https://mmocr.readthedocs.io/en/dev-1.x/user_guides/useful_tools.html#dataset-visualization-tool  # noqa: E501

    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register all modules in mmocr into the registries
    register_all_modules()

    dataset, visualizer = obtain_dataset_cfg(cfg)
    dataset = DATASETS.build(dataset)
    visualizer = VISUALIZERS.build(visualizer)

    visualizer.dataset_meta = dataset.metainfo
    progress_bar = mmengine.ProgressBar(len(dataset))
    for item in dataset:
        img = item['inputs'].permute(1, 2, 0).numpy()
        data_sample = item['data_samples'].numpy()
        img_path = osp.basename(item['data_samples'].img_path)
        out_file = osp.join(args.output_dir,
                            img_path) if args.output_dir is not None else None

        if img.ndim == 3 and img.shape[-1] == 3:
            img = img[..., [2, 1, 0]]  # bgr to rgb

        visualizer.add_datasample(
            name=osp.basename(img_path),
            image=img,
            data_sample=data_sample,
            draw_pred=False,
            show=not args.not_show,
            wait_time=args.show_interval,
            out_file=out_file)

        progress_bar.update()


def obtain_dataset_cfg(cfg: Config) -> Tuple:
    """Obtain dataset and visualizer from config. Two modes are supported:
    1. Model Config Mode:
        In this mode, the input config should be a complete model config, which
        includes a dataset within pipeline and a visualizer.
    2. Dataset Config Mode:
        In this mode, the input config should be a complete dataset config,
        which only includes basic dataset information, and it may does not
        contain a visualizer and dataset pipeline.

    Examples:
        Typically, the model config files are stored in
        `configs/textdet/dbnet/xxx.py` and should be looked like:
        >>> train_dataloader = dict(
        >>>     batch_size=16,
        >>>     num_workers=8,
        >>>     persistent_workers=True,
        >>>     sampler=dict(type='DefaultSampler', shuffle=True),
        >>>     dataset=ic15_det_train)

        while the dataset config files are stored in
        `configs/textdet/_base_/datasets/xxx.py` and should be like:
        >>> ic15_det_train = dict(
        >>>     type='OCRDataset',
        >>>     data_root=ic15_det_data_root,
        >>>     ann_file='textdet_train.json',
        >>>     filter_cfg=dict(filter_empty_gt=True, min_size=32),
        >>>     pipeline=None)

    Args:
        cfg (Config): Config object.

    Returns:
        Tuple: Tuple of (dataset, visualizer).
    """

    # Model config mode
    if 'train_dataloader' in cfg:
        dataset = cfg.train_dataloader.dataset
        visualizer = cfg.visualizer

        return dataset, visualizer

    # Dataset config mode
    default_visualizer = dict(
        type='TextDetLocalVisualizer',
        name='visualizer',
        vis_backends=[dict(type='LocalVisBackend')])

    default_det_pipeline = [
        dict(
            type='LoadImageFromFile',
            file_client_args=dict(backend='disk'),
            color_type='color_ignore_orientation'),
        dict(
            type='LoadOCRAnnotations',
            with_polygon=True,
            with_bbox=True,
            with_label=True,
        ),
        dict(
            type='PackTextDetInputs',
            meta_keys=('img_path', 'ori_shape', 'img_shape'))
    ]

    default_rec_pipeline = [
        dict(
            type='LoadImageFromFile',
            file_client_args=dict(backend='disk'),
            ignore_empty=True,
            min_size=2),
        dict(type='LoadOCRAnnotations', with_text=True),
        dict(
            type='PackTextRecogInputs',
            meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
    ]

    for key in cfg.keys():
        if key.endswith('train'):
            dataset = cfg[key]
            if 'det' in key.lower():
                visualizer = default_visualizer
                dataset['pipeline'] = default_det_pipeline if dataset[
                    'pipeline'] is None else dataset['pipeline']
            elif 'rec' in key.lower():
                default_visualizer['type'] = 'TextRecogLocalVisualizer'
                visualizer = default_visualizer
                dataset['pipeline'] = default_rec_pipeline if dataset[
                    'pipeline'] is None else dataset['pipeline']
            else:
                raise NotImplementedError(
                    'Dataset config mode only supports text detection and '
                    'recognition datasets yet. Please ensure the dataset '
                    'config contains "det" or "rec" in its key.')

            return dataset, visualizer

    raise ValueError(
        'Unexpected config file format. Please check your config '
        'file and try again. More details can be found in the docstring of '
        'obtain_dataset_cfg function. Or, you may visit the documentation via '
        'https://mmocr.readthedocs.io/en/dev-1.x/user_guides/useful_tools.html#dataset-visualization-tool'  # noqa: E501
    )


if __name__ == '__main__':
    main()
