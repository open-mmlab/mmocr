# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import warnings

from mmocr.datasets.preparers import DatasetPreparer
from mmocr.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preparing datasets used in MMOCR.')
    parser.add_argument(
        'datasets',
        help='A list of the dataset names that would like to prepare.',
        nargs='+')
    parser.add_argument(
        '--nproc', help='Number of processes to run', default=4, type=int)
    parser.add_argument(
        '--task',
        default='textdet',
        choices=['textdet', 'textrecog', 'textspotting', 'kie'],
        help='Task type. Options are det and rec.')
    parser.add_argument(
        '--dataset-zoo-path',
        default='./dataset_zoo',
        help='Path to dataset zoo config files.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    register_all_modules()
    for dataset in args.datasets:
        if not osp.isdir(osp.join(args.dataset_zoo_path, dataset)):
            warnings.warn(f'{dataset} is not supported yet. Please check '
                          'dataset zoo for supported datasets.')
            continue
        preparer = DatasetPreparer(
            cfg_path=args.dataset_zoo_path,
            dataset_name=dataset,
            task=args.task,
            nproc=args.nproc)
        preparer()


if __name__ == '__main__':
    main()
