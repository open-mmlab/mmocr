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
        help='Whether to dump the textrecog dataset to LMDB format, It\'s a '
        'shortcut to force the dataset to be dumped in lmdb format. '
        'Applicable when --task=textrecog')
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


def force_lmdb(cfg):
    """Force the dataset to be dumped in lmdb format.

    Args:
        cfg (Config): Config object.

    Returns:
        Config: Config object.
    """
    for split in ['train', 'val', 'test']:
        preparer_cfg = cfg.get(f'{split}_preparer')
        if preparer_cfg:
            if preparer_cfg.get('dumper') is None:
                raise ValueError(
                    f'{split} split does not come with a dumper, '
                    'so most likely the annotations are MMOCR-ready and do '
                    'not need any adaptation, and it '
                    'cannot be dumped in LMDB format.')
            preparer_cfg.dumper['type'] = 'LMDBDumper'

    cfg.config_generator['dataset_name'] = f'{cfg.dataset_name}_lmdb'
    for ann_list_key in ['train_anns', 'val_anns', 'test_anns']:
        if ann_list_key in cfg.config_generator:
            # It can be None when users want to clear out the default
            # value
            if not cfg.config_generator[ann_list_key]:
                continue
            ann_list = cfg.config_generator[ann_list_key]
            for ann_dict in ann_list:
                ann_dict['ann_file'] = (
                    osp.splitext(ann_dict['ann_file'])[0] + '.lmdb')
        else:
            if ann_list_key == 'train_anns':
                ann_list = [dict(ann_file='textrecog_train.lmdb')]
            elif ann_list_key == 'test_anns':
                ann_list = [dict(ann_file='textrecog_test.lmdb')]
            else:
                ann_list = []
        cfg.config_generator[ann_list_key] = ann_list
    return cfg


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
        if args.lmdb:
            cfg = force_lmdb(cfg)
        preparer = DatasetPreparer.from_file(cfg)
        preparer.run(args.splits)


if __name__ == '__main__':
    main()
