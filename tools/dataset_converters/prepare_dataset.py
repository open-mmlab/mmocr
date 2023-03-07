# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import time
import warnings

from mmengine import Config

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
        '--splits',
        default=['train', 'test', 'val'],
        help='A list of the split that would like to prepare.',
        nargs='+')
    parser.add_argument(
        '--lmdb',
        action='store_true',
        default=False,
        help='Whether to dump the textrecog dataset to LMDB format, '
        'applicable when --task=textrecog')
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


def parse_meta(task: str, meta_path: str) -> None:
    """Parse meta file.

    Args:
        cfg_path (str): Path to meta file.
    """
    try:
        meta = Config.fromfile(meta_path)
    except FileNotFoundError:
        return
    assert task in meta['Data']['Tasks'], \
        f'Task {task} not supported!'
    # License related
    if meta['Data']['License']['Type']:
        print(f"\033[1;33;40mDataset Name: {meta['Name']}")
        print(f"License Type: {meta['Data']['License']['Type']}")
        print(f"License Link: {meta['Data']['License']['Link']}")
        print(f"BibTeX: {meta['Paper']['BibTeX']}\033[0m")
        print('\033[1;31;43mMMOCR does not own the dataset. Using this '
              'dataset you must accept the license provided by the owners, '
              'and cite the corresponding papers appropriately.')
        print('If you do not agree with the above license, please cancel '
              'the progress immediately by pressing ctrl+c. Otherwise, '
              'you are deemed to accept the terms and conditions.\033[0m')
        for i in range(5):
            print(f'{5-i}...')
            time.sleep(1)


def main():
    args = parse_args()
    if args.lmdb and args.task != 'textrecog':
        raise ValueError('--lmdb only works with --task=textrecog')
    for dataset in args.datasets:
        if not osp.isdir(osp.join(args.dataset_zoo_path, dataset)):
            warnings.warn(f'{dataset} is not supported yet. Please check '
                          'dataset zoo for supported datasets.')
            continue
        meta_path = osp.join(args.dataset_zoo_path, dataset, 'metafile.yml')
        parse_meta(args.task, meta_path)
        cfg_path = osp.join(args.dataset_zoo_path, dataset, args.task + '.py')
        cfg = Config.fromfile(cfg_path)
        if args.overwrite_cfg and cfg.get('config_generator',
                                          None) is not None:
            cfg.config_generator.overwrite_cfg = args.overwrite_cfg
        cfg.nproc = args.nproc
        cfg.task = args.task
        cfg.dataset_name = dataset
        cfg.lmdb = args.lmdb
        preparer = DatasetPreparer.from_file(cfg)
        preparer.run(args.splits)


if __name__ == '__main__':
    main()
