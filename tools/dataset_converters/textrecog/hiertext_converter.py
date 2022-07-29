# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import math
import os.path as osp
from functools import partial

import mmcv
import numpy as np
from shapely.geometry import Polygon

from mmocr.utils.fileio import list_to_file


def seg2bbox(seg):
    """Convert segmentation to bbox.

    Args:
        seg (list(int | float)): A set of coordinates
    """
    if len(seg) == 4:
        min_x = min(seg[0], seg[2], seg[4], seg[6])
        max_x = max(seg[0], seg[2], seg[4], seg[6])
        min_y = min(seg[1], seg[3], seg[5], seg[7])
        max_y = max(seg[1], seg[3], seg[5], seg[7])
    else:
        seg = np.array(seg).reshape(-1, 2)
        polygon = Polygon(seg)
        min_x, min_y, max_x, max_y = polygon.bounds
    bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
    return bbox


def process_level(
    src_img,
    annotation,
    dst_image_root,
    ignore_image_root,
    preserve_vertical,
    split,
    format,
    para_idx,
    img_idx,
    line_idx,
    word_idx=None,
):
    vertices = annotation['vertices']
    text_label = annotation['text']
    segmentation = [i for j in vertices for i in j]
    x, y, w, h = seg2bbox(segmentation)
    x, y = max(0, math.floor(x)), max(0, math.floor(y))
    w, h = math.ceil(w), math.ceil(h)
    dst_img = src_img[y:y + h, x:x + w]
    if word_idx:
        dst_img_name = f'img_{img_idx}_{para_idx}_{line_idx}_{word_idx}.jpg'
    else:
        dst_img_name = f'img_{img_idx}_{para_idx}_{line_idx}.jpg'
    if not preserve_vertical and h / w > 2 and split == 'train':
        dst_img_path = osp.join(ignore_image_root, dst_img_name)
        mmcv.imwrite(dst_img, dst_img_path)
        return None

    dst_img_path = osp.join(dst_image_root, dst_img_name)
    mmcv.imwrite(dst_img, dst_img_path)

    if format == 'txt':
        label = (f'{osp.basename(dst_image_root)}/{dst_img_name}'
                 f' {text_label}')
    elif format == 'jsonl':
        label = json.dumps({
            'filename': f'{osp.basename(dst_image_root)}/{dst_img_name}',
            'text': text_label
        })
    else:
        raise NotImplementedError
    return label


def process_img(args, src_image_root, dst_image_root, ignore_image_root, level,
                preserve_vertical, split, format):
    # Dirty hack for multi-processing
    img_idx, img_annos = args
    src_img = mmcv.imread(
        osp.join(src_image_root, img_annos['image_id'] + '.jpg'))
    labels = []
    for para_idx, paragraph in enumerate(img_annos['paragraphs']):
        for line_idx, line in enumerate(paragraph['lines']):
            if level == 'line':
                # Ignore illegible words
                if line['legible']:

                    label = process_level(src_img, line, dst_image_root,
                                          ignore_image_root, preserve_vertical,
                                          split, format, para_idx, img_idx,
                                          line_idx)
                    if label is not None:
                        labels.append(label)
            elif level == 'word':
                for word_idx, word in enumerate(line['words']):
                    if not word['legible']:
                        continue
                    label = process_level(src_img, word, dst_image_root,
                                          ignore_image_root, preserve_vertical,
                                          split, format, para_idx, img_idx,
                                          line_idx, word_idx)
                    if label is not None:
                        labels.append(label)
    return labels


def convert_hiertext(
    root_path,
    split,
    level,
    preserve_vertical,
    format,
    nproc,
):
    """Collect the annotation information and crop the images.

    The annotation format is as the following:
    {
        "info": {
            "date": "release date",
            "version": "current version"
        },
        "annotations": [  // List of dictionaries, one for each image.
            {
            "image_id": "the filename of corresponding image.",
            "image_width": image_width,  // (int) The image width.
            "image_height": image_height, // (int) The image height.
            "paragraphs": [  // List of paragraphs.
                {
                "vertices": [[x1, y1], [x2, y2],...,[xn, yn]]
                "legible": true
                "lines": [
                    {
                    "vertices": [[x1, y1], [x2, y2],...,[x4, y4]]
                    "text": L
                    "legible": true,
                    "handwritten": false
                    "vertical": false,
                    "words": [
                        {
                        "vertices": [[x1, y1], [x2, y2],...,[xm, ym]]
                        "text": "the text content of this word",
                        "legible": true
                        "handwritten": false,
                        "vertical": false,
                        }, ...
                    ]
                    }, ...
                ]
                }, ...
            ]
            }, ...
        ]
        }

    Args:
        root_path (str): Root path to the dataset
        split (str): Dataset split, which should be 'train' or 'val'
        level (str): Crop word or line level instances
        preserve_vertical (bool): Whether to preserve vertical texts
        format (str): Annotation format, should be either 'jsonl' or 'txt'
        nproc (int): Number of processes

    Returns:
        img_info (dict): The dict of the img and annotation information
    """

    annotation_path = osp.join(root_path, 'annotations/' + split + '.jsonl')
    if not osp.exists(annotation_path):
        raise Exception(
            f'{annotation_path} not exists, please check and try again.')

    annotation = json.load(open(annotation_path))['annotations']
    # outputs
    dst_label_file = osp.join(root_path, f'{split}_label.{format}')
    dst_image_root = osp.join(root_path, 'crops', split)
    ignore_image_root = osp.join(root_path, 'ignores', split)
    src_image_root = osp.join(root_path, 'imgs', split)
    mmcv.mkdir_or_exist(dst_image_root)
    mmcv.mkdir_or_exist(ignore_image_root)

    process_img_with_path = partial(
        process_img,
        src_image_root=src_image_root,
        dst_image_root=dst_image_root,
        ignore_image_root=ignore_image_root,
        level=level,
        preserve_vertical=preserve_vertical,
        split=split,
        format=format)
    tasks = []
    for img_idx, img_info in enumerate(annotation):
        tasks.append((img_idx, img_info))
    labels_list = mmcv.track_parallel_progress(
        process_img_with_path, tasks, keep_order=True, nproc=nproc)

    final_labels = []
    for label_list in labels_list:
        final_labels += label_list
    list_to_file(dst_label_file, final_labels)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and validation set of HierText')
    parser.add_argument('root_path', help='Root dir path of HierText')
    parser.add_argument(
        '--nproc', default=1, type=int, help='Number of processes')
    parser.add_argument(
        '--preserve-vertical',
        help='Preserve samples containing vertical texts',
        action='store_true')
    parser.add_argument(
        '--level',
        default='word',
        help='Crop word or line level instance',
        choices=['word', 'line'])
    parser.add_argument(
        '--format',
        default='jsonl',
        help='Use jsonl or string to format annotations',
        choices=['jsonl', 'txt'])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path
    print('Processing training set...')
    convert_hiertext(
        root_path=root_path,
        split='train',
        level=args.level,
        preserve_vertical=args.preserve_vertical,
        format=args.format,
        nproc=args.nproc)
    print('Processing validation set...')
    convert_hiertext(
        root_path=root_path,
        split='val',
        level=args.level,
        preserve_vertical=args.preserve_vertical,
        format=args.format,
        nproc=args.nproc)
    print('Finish')


if __name__ == '__main__':
    main()
