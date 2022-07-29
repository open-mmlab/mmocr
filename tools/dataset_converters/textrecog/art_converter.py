# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import math
import os.path as osp

import mmcv

from mmocr.utils.fileio import list_to_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and validation set of ArT ')
    parser.add_argument('root_path', help='Root dir path of ArT')
    parser.add_argument(
        '--val-ratio', help='Split ratio for val set', default=0.0, type=float)
    parser.add_argument(
        '--nproc', default=1, type=int, help='Number of processes')
    parser.add_argument(
        '--format',
        default='jsonl',
        help='Use jsonl or string to format annotations',
        choices=['jsonl', 'txt'])
    args = parser.parse_args()
    return args


def convert_art(root_path, split, ratio, format):
    """Collect the annotation information and crop the images.

    The annotation format is as the following:
    {
        "gt_2836_0": [
            {
                "transcription": "URDER",
                "points": [
                    [25, 51],
                    [0, 2],
                    [21, 0],
                    [42, 43]
                ],
                "language": "Latin",
                "illegibility": false
            }
        ], ...
    }


    Args:
        root_path (str): The root path of the dataset
        split (str): The split of dataset. Namely: training or val
        ratio (float): Split ratio for val set
        format (str): Annotation format, whether be txt or jsonl

    Returns:
        img_info (dict): The dict of the img and annotation information
    """

    annotation_path = osp.join(root_path,
                               'annotations/train_task2_labels.json')
    if not osp.exists(annotation_path):
        raise Exception(
            f'{annotation_path} not exists, please check and try again.')

    annotation = mmcv.load(annotation_path)
    # outputs
    dst_label_file = osp.join(root_path, f'{split}_label.{format}')

    img_prefixes = annotation.keys()

    trn_files, val_files = [], []
    if ratio > 0:
        for i, file in enumerate(img_prefixes):
            if i % math.floor(1 / ratio):
                trn_files.append(file)
            else:
                val_files.append(file)
    else:
        trn_files, val_files = img_prefixes, []
    print(f'training #{len(trn_files)}, val #{len(val_files)}')

    if split == 'train':
        img_prefixes = trn_files
    elif split == 'val':
        img_prefixes = val_files
    else:
        raise NotImplementedError

    labels = []
    for prefix in img_prefixes:
        text_label = annotation[prefix][0]['transcription']
        dst_img_name = prefix + '.jpg'

        if format == 'txt':
            labels.append(f'crops/{dst_img_name}' f' {text_label}')
        elif format == 'jsonl':
            labels.append(
                json.dumps(
                    {
                        'filename': f'crops/{dst_img_name}',
                        'text': text_label
                    },
                    ensure_ascii=False))

    list_to_file(dst_label_file, labels)


def main():
    args = parse_args()
    root_path = args.root_path
    print('Processing training set...')
    convert_art(
        root_path=root_path,
        split='train',
        ratio=args.val_ratio,
        format=args.format)
    if args.val_ratio > 0:
        print('Processing validation set...')
        convert_art(
            root_path=root_path,
            split='val',
            ratio=args.val_ratio,
            format=args.format)
    print('Finish')


if __name__ == '__main__':
    main()
