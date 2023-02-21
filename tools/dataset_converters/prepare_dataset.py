# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import warnings

from mmocr.datasets.preparers import DatasetPreparer


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
        help='Task type. Options are "textdet", "textrecog", "textspotting"'
        ' and "kie".')
    parser.add_argument(
        '--lmdb',
        action='store_true',
        default=False,
        help='Whether to dump the textrecog dataset to LMDB format.'
        ' Only when --task=textrecog, this argument works')
    parser.add_argument(
        '--overwrite-cfg',
        action='store_true',
        default=False,
        help='Whether to overwrite the dataset config file if it already'
        ' exists. If not specified, Dataset Preparer will not generate'
        ' new config for datasets whose configs are already in base.')
    parser.add_argument(
        '--dataset-zoo-path',
        default='./dataset_zoo',
        help='Path to dataset zoo config files.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.lmdb and args.task != 'textrecog':
        raise ValueError('Only textrecog task can use --lmdb.')

    for dataset in args.datasets:
        if not osp.isdir(osp.join(args.dataset_zoo_path, dataset)):
            warnings.warn(f'{dataset} is not supported yet. Please check '
                          'dataset zoo for supported datasets.')
            continue
        preparer = DatasetPreparer(
            cfg_path=args.dataset_zoo_path,
            dataset_name=dataset,
            task=args.task,
            nproc=args.nproc,
            dump_to_lmdb=args.lmdb,
            overwrite_cfg=args.overwrite_cfg)
        preparer()


if __name__ == '__main__':
    main()
