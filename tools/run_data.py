#!/usr/bin/env python
import argparse
import copy

import mmcv
from mmcv import Config, DictAction

from mmdet.datasets import build_dataloader
from mmocr.datasets import build_dataset

assert build_dataset is not None


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector.')
    parser.add_argument('config', help='Train config file path.')
    parser.add_argument('--work-dir', help='The dir to save logs and models.')
    parser.add_argument(
        '--load-from', help='The checkpoint file to load from.')
    parser.add_argument(
        '--resume-from', help='The checkpoint file to resume from.')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Whether not to evaluate the checkpoint during training.')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='Number of gpus to use '
        '(only applicable to non-distributed training).')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training).')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be of the form of either '
        'key="[a,b]" or key=a,b .The argument also allows nested list/tuple '
        'values, e.g. key="[(a,b),(c,d)]". Note that the quotation marks '
        'are necessary and that no white space is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Options for job launcher.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--mc-config',
        type=str,
        default='',
        help='Memory cache config for image loading speed-up during training.')

    args = parser.parse_args()

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # update mc config
    if args.mc_config:
        mc = Config.fromfile(args.mc_config)
        if isinstance(cfg.data.train, list):
            for i in range(len(cfg.data.train)):
                cfg.data.train[i].pipeline[0].update(
                    file_client_args=mc['mc_file_client_args'])
        else:
            cfg.data.train.pipeline[0].update(
                file_client_args=mc['mc_file_client_args'])

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    # prepare data loaders
    if 'imgs_per_gpu' in cfg.data:
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            1,
            dist=False,
            seed=None) for ds in datasets
    ]

    for i in range(100):
        print(f'\n= {i+1}-th pass:', flush=True)
        for j, dataloader in enumerate(data_loaders):
            print(f'\n{j+1}/{len(data_loaders)}-th dataloader:', flush=True)
            prog_bar = mmcv.ProgressBar(len(dataloader))
            for _ in dataloader:
                prog_bar.update()


if __name__ == '__main__':
    main()
