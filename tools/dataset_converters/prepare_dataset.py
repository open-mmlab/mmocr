# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

from mmocr.registry import TASK_UTILS
from mmocr.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preparing datasets used in MMOCR.')
    parser.add_argument('dataset', help='Name of the dataset')
    parser.add_argument(
        '--nproc', help='Number of processes to run', default=4)
    parser.add_argument('--task', default='det', choices=['det', 'rec'])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not args.dataset + '.yml' in os.listdir('./dataset_zoo/'):
        raise NotImplementedError(f'{args.dataset} is not supported yet. '
                                  'Please check dataset zoo for supported '
                                  'datasets.')
    register_all_modules()
    preparer = TASK_UTILS.build(
        dict(
            type=f'{args.dataset.upper()}Preparer',
            dataset_name=args.dataset,
            nproc=args.nproc,
            task=args.task))
    preparer.process()


if __name__ == '__main__':
    main()
