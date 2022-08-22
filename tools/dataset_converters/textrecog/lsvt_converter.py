# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import math
import os.path as osp
from functools import partial

import mmcv
import mmengine

from mmocr.utils.fileio import list_to_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and validation set of LSVT ')
    parser.add_argument('root_path', help='Root dir path of LSVT')
    parser.add_argument(
        '--val-ratio', help='Split ratio for val set', default=0.0, type=float)
    parser.add_argument(
        '--nproc', default=1, type=int, help='Number of processes')
    parser.add_argument(
        '--preserve-vertical',
        help='Preserve samples containing vertical texts',
        action='store_true')
    parser.add_argument(
        '--format',
        default='jsonl',
        help='Use jsonl or string to format annotations',
        choices=['jsonl', 'txt'])
    args = parser.parse_args()
    return args


def process_img(args, dst_image_root, ignore_image_root, preserve_vertical,
                split, format):
    # Dirty hack for multi-processing
    img_idx, img_info, anns = args
    src_img = mmcv.imread(img_info['file_name'])
    labels = []
    for ann_idx, ann in enumerate(anns):
        segmentation = []
        for x, y in ann['points']:
            segmentation.append(max(0, x))
            segmentation.append(max(0, y))
        xs, ys = segmentation[::2], segmentation[1::2]
        x, y = min(xs), min(ys)
        w, h = max(xs) - x, max(ys) - y
        text_label = ann['transcription']

        dst_img = src_img[y:y + h, x:x + w]
        dst_img_name = f'img_{img_idx}_{ann_idx}.jpg'

        if not preserve_vertical and h / w > 2 and split == 'train':
            dst_img_path = osp.join(ignore_image_root, dst_img_name)
            mmcv.imwrite(dst_img, dst_img_path)
            continue

        dst_img_path = osp.join(dst_image_root, dst_img_name)
        mmcv.imwrite(dst_img, dst_img_path)
        if format == 'txt':
            labels.append(f'{osp.basename(dst_image_root)}/{dst_img_name}'
                          f' {text_label}')
        elif format == 'jsonl':
            labels.append(
                json.dumps({
                    'filename':
                    f'{osp.basename(dst_image_root)}/{dst_img_name}',
                    'text': text_label
                }))
        else:
            raise NotImplementedError

    return labels


def convert_lsvt(root_path,
                 split,
                 ratio,
                 preserve_vertical,
                 format,
                 nproc,
                 img_start_idx=0):
    """Collect the annotation information and crop the images.

    The annotation format is as the following:
    [
        {'gt_1234': # 'gt_1234' is file name
            [
                {
                    'transcription': '一站式购物中心',
                    'points': [[45, 272], [215, 273], [212, 296], [45, 290]]
                    'illegibility': False
                }, ...
            ]
        }
    ]


    Args:
        root_path (str): The root path of the dataset
        split (str): The split of dataset. Namely: training or val
        ratio (float): Split ratio for val set
        preserve_vertical (bool): Whether to preserve vertical texts
        format (str): Annotation format, whether be txt or jsonl
        nproc (int): The number of process to collect annotations
        img_start_idx (int): Index of start image

    Returns:
        img_info (dict): The dict of the img and annotation information
    """

    annotation_path = osp.join(root_path, 'annotations/train_full_labels.json')
    if not osp.exists(annotation_path):
        raise Exception(
            f'{annotation_path} not exists, please check and try again.')

    annotation = mmengine.load(annotation_path)
    # outputs
    dst_label_file = osp.join(root_path, f'{split}_label.{format}')
    dst_image_root = osp.join(root_path, 'crops', split)
    ignore_image_root = osp.join(root_path, 'ignores', split)
    src_image_root = osp.join(root_path, 'imgs')
    mmengine.mkdir_or_exist(dst_image_root)
    mmengine.mkdir_or_exist(ignore_image_root)

    process_img_with_path = partial(
        process_img,
        dst_image_root=dst_image_root,
        ignore_image_root=ignore_image_root,
        preserve_vertical=preserve_vertical,
        split=split,
        format=format)

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

    tasks = []
    idx = 0
    for img_idx, prefix in enumerate(img_prefixes):
        img_file = osp.join(src_image_root, prefix + '.jpg')
        img_info = {'file_name': img_file}
        # Skip not exist images
        if not osp.exists(img_file):
            continue
        tasks.append((img_idx + img_start_idx, img_info, annotation[prefix]))
        idx = idx + 1

    labels_list = mmengine.track_parallel_progress(
        process_img_with_path, tasks, keep_order=True, nproc=nproc)
    final_labels = []
    for label_list in labels_list:
        final_labels += label_list
    list_to_file(dst_label_file, final_labels)

    return idx


def main():
    args = parse_args()
    root_path = args.root_path
    print('Processing training set...')
    num_train_imgs = convert_lsvt(
        root_path=root_path,
        split='train',
        ratio=args.val_ratio,
        preserve_vertical=args.preserve_vertical,
        format=args.format,
        nproc=args.nproc)
    if args.val_ratio > 0:
        print('Processing validation set...')
        convert_lsvt(
            root_path=root_path,
            split='val',
            ratio=args.val_ratio,
            preserve_vertical=args.preserve_vertical,
            format=args.format,
            nproc=args.nproc,
            img_start_idx=num_train_imgs)
    print('Finish')


if __name__ == '__main__':
    main()
