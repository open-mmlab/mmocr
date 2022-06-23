# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import math
import os
import os.path as osp

import cv2
import mmcv
from PIL import Image

from mmocr.datasets.pipelines.crop import crop_img
from mmocr.utils.fileio import list_to_file


def collect_files(img_dir, gt_dir, ratio):
    """Collect all images and their corresponding groundtruth files.
    Args:
        img_dir (str): The image directory
        gt_dir (str): The groundtruth directory
        ratio (float): Split ratio for val set

    Returns:
        files (list): The list of tuples (img_file, groundtruth_file)
    """
    assert isinstance(img_dir, str)
    assert img_dir
    assert isinstance(gt_dir, str)
    assert gt_dir
    assert isinstance(ratio, float)
    assert ratio < 1.0, 'val_ratio should be a float between 0.0 to 1.0'

    ann_list, imgs_list = [], []
    for ann_file in os.listdir(gt_dir):
        img_file = osp.join(img_dir, ann_file.replace('txt', 'jpg'))
        # This dataset contains some images obtained from .gif,
        # which cannot be loaded by mmcv.imread(), convert them
        # to RGB mode.
        try:
            if mmcv.imread(img_file) is None:
                print(f'Convert {img_file} to RGB mode.')
                img = Image.open(img_file)
                img = img.convert('RGB')
                img.save(img_file)
        except cv2.error:
            print(f'Skip broken img {img_file}')
            continue

        ann_list.append(osp.join(gt_dir, ann_file))
        imgs_list.append(img_file)

    all_files = list(zip(imgs_list, ann_list))
    assert len(all_files), f'No images found in {img_dir}'
    print(f'Loaded {len(all_files)} images from {img_dir}')

    trn_files, val_files = [], []
    if ratio > 0:
        for i, file in enumerate(all_files):
            if i % math.floor(1 / ratio):
                trn_files.append(file)
            else:
                val_files.append(file)
    else:
        trn_files, val_files = all_files, []

    print(f'training #{len(trn_files)}, val #{len(val_files)}')

    return trn_files, val_files


def collect_annotations(files, nproc=1):
    """Collect the annotation information.
    Args:
        files (list): The list of tuples (image_file, groundtruth_file)
        nproc (int): The number of process to collect annotations

    Returns:
        images (list): The list of image information dicts
    """
    assert isinstance(files, list)
    assert isinstance(nproc, int)

    if nproc > 1:
        images = mmcv.track_parallel_progress(
            load_img_info, files, nproc=nproc)
    else:
        images = mmcv.track_progress(load_img_info, files)

    return images


def load_img_info(files):
    """Load the information of one image.
    Args:
        files (tuple): The tuple of (img_file, groundtruth_file)

    Returns:
        img_info (dict): The dict of the img and annotation information
    """
    assert isinstance(files, tuple)

    img_file, gt_file = files
    assert osp.basename(gt_file).split('.')[0] == osp.basename(img_file).split(
        '.')[0]
    # read imgs while ignoring orientations
    img = mmcv.imread(img_file)

    img_info = dict(
        file_name=osp.join(osp.basename(img_file)),
        height=img.shape[0],
        width=img.shape[1],
        segm_file=osp.join(osp.basename(gt_file)))

    if osp.splitext(gt_file)[1] == '.txt':
        img_info = load_txt_info(gt_file, img_info)
    else:
        raise NotImplementedError

    return img_info


def load_txt_info(gt_file, img_info):
    """Collect the annotation information.

    The annotation format is as the following:
    x1,y1,x2,y2,x3,y3,x4,y4,text

    45.45,226.83,11.87,181.79,183.84,13.1,233.79,49.95,时尚袋袋
    345.98,311.18,345.98,347.21,462.26,347.21,462.26,311.18,73774
    462.26,292.34,461.44,299.71,502.39,299.71,502.39,292.34,73/74/737

    Args:
        gt_file (str): The path to ground-truth
        img_info (dict): The dict of the img and annotation information

    Returns:
        img_info (dict): The dict of the img and annotation information
    """

    anno_info = []
    with open(gt_file) as f:
        lines = f.readlines()
    for line in lines:
        points = line.split(',')[0:8]
        word = line.split(',')[8].rstrip('\n')
        if word == '###':
            continue
        bbox = [math.floor(float(pt)) for pt in points]
        anno = dict(bbox=bbox, word=word)
        anno_info.append(anno)

    img_info.update(anno_info=anno_info)

    return img_info


def generate_ann(root_path, split, image_infos, preserve_vertical, format):
    """Generate cropped annotations and label txt file.

    Args:
        root_path (str): The root path of the dataset
        split (str): The split of dataset. Namely: training or test
        image_infos (list[dict]): A list of dicts of the img and
            annotation information
        preserve_vertical (bool): Whether to preserve vertical texts
        format (str): Annotation format, whether be txt or jsonl
    """
    print('Cropping images...')
    dst_image_root = osp.join(root_path, 'crops', split)
    ignore_image_root = osp.join(root_path, 'ignores', split)
    if split == 'training':
        dst_label_file = osp.join(root_path, f'train_label.{format}')
    elif split == 'val':
        dst_label_file = osp.join(root_path, f'val_label.{format}')
    mmcv.mkdir_or_exist(dst_image_root)
    mmcv.mkdir_or_exist(ignore_image_root)

    lines = []
    for image_info in image_infos:
        index = 1
        src_img_path = osp.join(root_path, 'imgs', image_info['file_name'])
        image = mmcv.imread(src_img_path)
        src_img_root = image_info['file_name'].split('.')[0]

        for anno in image_info['anno_info']:
            word = anno['word']
            dst_img = crop_img(image, anno['bbox'], 0, 0)
            h, w, _ = dst_img.shape

            dst_img_name = f'{src_img_root}_{index}.png'
            index += 1
            # Skip invalid annotations
            if min(dst_img.shape) == 0:
                continue
            # Skip vertical texts
            if not preserve_vertical and h / w > 2 and split == 'training':
                dst_img_path = osp.join(ignore_image_root, dst_img_name)
                mmcv.imwrite(dst_img, dst_img_path)
                continue

            dst_img_path = osp.join(dst_image_root, dst_img_name)
            mmcv.imwrite(dst_img, dst_img_path)
            if format == 'txt':
                lines.append(f'{osp.basename(dst_image_root)}/{dst_img_name} '
                             f'{word}')
            elif format == 'jsonl':
                lines.append(
                    json.dumps(
                        {
                            'filename':
                            f'{osp.basename(dst_image_root)}/{dst_img_name}',
                            'text': word
                        },
                        ensure_ascii=False))
            else:
                raise NotImplementedError

    list_to_file(dst_label_file, lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and val set of MTWI.')
    parser.add_argument('root_path', help='Root dir path of MTWI')
    parser.add_argument(
        '--val-ratio', help='Split ratio for val set', default=0.0, type=float)
    parser.add_argument(
        '--preserve-vertical',
        help='Preserve samples containing vertical texts',
        action='store_true')
    parser.add_argument(
        '--format',
        default='jsonl',
        help='Use jsonl or string to format annotations',
        choices=['jsonl', 'txt'])
    parser.add_argument(
        '--nproc', default=1, type=int, help='Number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path
    ratio = args.val_ratio

    trn_files, val_files = collect_files(
        osp.join(root_path, 'imgs'), osp.join(root_path, 'annotations'), ratio)

    # Train set
    trn_infos = collect_annotations(trn_files, nproc=args.nproc)
    with mmcv.Timer(
            print_tmpl='It takes {}s to convert MTWI Training annotation'):
        generate_ann(root_path, 'training', trn_infos, args.preserve_vertical,
                     args.format)

    # Val set
    if len(val_files) > 0:
        val_infos = collect_annotations(val_files, nproc=args.nproc)
        with mmcv.Timer(
                print_tmpl='It takes {}s to convert MTWI Val annotation'):
            generate_ann(root_path, 'val', val_infos, args.preserve_vertical,
                         args.format)


if __name__ == '__main__':
    main()
