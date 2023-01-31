# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math
import os.path as osp

import mmengine
from mmocr.utils import dump_ocr_data


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and validation set of ArT ')
    parser.add_argument('root_path', help='Root dir path of ArT')
    parser.add_argument(
        '--val-ratio', help='Split ratio for val set', default=0.0, type=float)
    parser.add_argument(
        '--nproc', default=1, type=int, help='Number of processes')
    args = parser.parse_args()
    return args


def convert_art(root_path, split, ratio):
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

    Returns:
        img_info (dict): The dict of the img and annotation information
    """

    annotation_path = osp.join(root_path,
                               'annotations/train_task2_labels.json')
    if not osp.exists(annotation_path):
        raise Exception(
            f'{annotation_path} not exists, please check and try again.')

    annotation = mmengine.load(annotation_path)
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

    img_info = []
    for prefix in img_prefixes:
        text_label = annotation[prefix][0]['transcription']
        dst_img_name = prefix + '.jpg'

        img_info.append({
            'file_name': dst_img_name,
            'anno_info': [{
                'text': text_label
            }]
        })

    ensure_ascii = dict(ensure_ascii=False)
    dump_ocr_data(img_info, osp.join(root_path, f'{split.lower()}_label.json'),
                  'textrecog', **ensure_ascii)


def main():
    args = parse_args()
    root_path = args.root_path
    print('Processing training set...')
    convert_art(root_path=root_path, split='train', ratio=args.val_ratio)
    if args.val_ratio > 0:
        print('Processing validation set...')
        convert_art(root_path=root_path, split='val', ratio=args.val_ratio)
    print('Finish')


if __name__ == '__main__':
    main()
