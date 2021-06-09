#!/usr/bin/env python
"""This file is for benchmark data loading process. It can also be used to
refresh the memcached cache. The command line to run this file is:

$ python -m cProfile -o program.prof tools/analysis/benchmark_processing.py
configs/task/method/[config filename]

Note: When debugging, the `workers_per_gpu` in the config should be set to 0
during benchmark.

It use cProfile to record cpu running time and output to program.prof
To visualize cProfile output program.prof, use Snakeviz and run:
$ snakeviz program.prof
"""
import argparse

import mmcv
from mmcv import Config

from mmdet.datasets import build_dataloader
from mmocr.datasets import build_dataset

assert build_dataset is not None


def main():
    parser = argparse.ArgumentParser(description='Benchmark data loading')
    parser.add_argument('config', help='Train config file path.')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)

    dataset = build_dataset(cfg.data.train)

    # prepare data loaders
    if 'imgs_per_gpu' in cfg.data:
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loader = build_dataloader(
        dataset,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        1,
        dist=False,
        seed=None)

    # Start progress bar after first 5 batches
    prog_bar = mmcv.ProgressBar(
        len(dataset) - 5 * cfg.data.samples_per_gpu, start=False)
    for i, data in enumerate(data_loader):
        if i == 5:
            prog_bar.start()
        for _ in range(len(data['img'])):
            if i < 5:
                continue
            prog_bar.update()


if __name__ == '__main__':
    main()
