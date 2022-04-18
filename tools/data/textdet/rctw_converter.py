# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math
import os
import os.path as osp

import mmcv

from mmocr.utils import convert_annotations


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
        ann_list.append(osp.join(gt_dir, ann_file))
        imgs_list.append(osp.join(img_dir, ann_file.replace('txt', 'jpg')))

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
    x1, y1, x2, y2, x3, y3, x4, y4, difficult, text

    390,902,1856,902,1856,1225,390,1225,0,"金氏眼镜"
    1875,1170,2149,1170,2149,1245,1875,1245,0,"创于1989"
    2054,1277,2190,1277,2190,1323,2054,1323,0,"城建店"

    Args:
        gt_file (str): The path to ground-truth
        img_info (dict): The dict of the img and annotation information

    Returns:
        img_info (dict): The dict of the img and annotation information
    """

    anno_info = []
    with open(gt_file, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    for line in lines:
        points = line.split(',')[0:8]
        word = line.split(',')[9].rstrip('\n').strip('"')
        difficult = 1 if line.split(',')[8] != '0' else 0
        segmentation = [int(pt) for pt in points]
        x = max(0, min(segmentation[0::2]))
        y = max(0, min(segmentation[1::2]))
        w = abs(max(segmentation[0::2]) - x)
        h = abs(max(segmentation[1::2]) - y)
        bbox = [x, y, w, h]

        if word == '###' or difficult == 1:
            iscrowd = 1
        else:
            iscrowd = 0

        anno = dict(
            iscrowd=iscrowd,
            category_id=1,
            bbox=bbox,
            area=w * h,
            segmentation=[segmentation])
        anno_info.append(anno)

    img_info.update(anno_info=anno_info)

    return img_info


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and val set of RCTW.')
    parser.add_argument('root_path', help='Root dir path of RCTW')
    parser.add_argument(
        '--val-ratio', help='Split ratio for val set', default=0.0, type=float)
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
    with mmcv.Timer(
            print_tmpl='It takes {}s to convert RCTW Training annotation'):
        trn_infos = collect_annotations(trn_files, nproc=args.nproc)
        convert_annotations(trn_infos,
                            osp.join(root_path, 'instances_training.json'))

    # Val set
    if len(val_files) > 0:
        with mmcv.Timer(
                print_tmpl='It takes {}s to convert RCTW Val annotation'):
            val_infos = collect_annotations(val_files, nproc=args.nproc)
            convert_annotations(val_infos,
                                osp.join(root_path, 'instances_val.json'))


if __name__ == '__main__':
    main()
