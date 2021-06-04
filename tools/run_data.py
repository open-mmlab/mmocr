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
