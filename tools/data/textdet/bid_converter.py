# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import mmcv

from mmocr.utils import convert_annotations


def collect_files(img_dir, gt_dir):
    """Collect all images and their corresponding groundtruth files.

    Args:
        img_dir (str): The image directory
        gt_dir (str): The groundtruth directory

    Returns:
        files (list): The list of tuples (img_file, groundtruth_file)
    """
    assert isinstance(img_dir, str)
    assert img_dir
    assert isinstance(gt_dir, str)
    assert gt_dir

    ann_list, imgs_list = [], []
    for img_file in os.listdir(img_dir):
        ann_file = img_file.split('_')[0] + '_gt_ocr.txt'
        ann_list.append(osp.join(gt_dir, ann_file))
        imgs_list.append(osp.join(img_dir, img_file))

    files = list(zip(imgs_list, ann_list))
    assert len(files), f'No images found in {img_dir}'
    print(f'Loaded {len(files)} images from {img_dir}')

    return files


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
    assert osp.basename(gt_file).split('_')[0] == osp.basename(gt_file).split(
        '_')[0]
    # read imgs while ignoring orientations
    img = mmcv.imread(img_file, 'unchanged')

    img_info = dict(
        file_name=osp.basename(img_file),
        height=img.shape[0],
        width=img.shape[1],
        segm_file=osp.basename(gt_file))

    if osp.splitext(gt_file)[1] == '.txt':
        img_info = load_txt_info(gt_file, img_info)
    else:
        raise NotImplementedError

    return img_info


def load_txt_info(gt_file, img_info):
    """Collect the annotation information.

    The annotation format is as the following:
    x, y, w, h, text
    977, 152, 16, 49, NOME
    962, 143, 12, 323, APPINHANESI BLAZEK PASSOTTO
    906, 446, 12, 94, 206940361
    905, 641, 12, 44, SPTC

    Args:
        gt_file (str): The path to ground-truth
        img_info (dict): The dict of the img and annotation information

    Returns:
        img_info (dict): The dict of the img and annotation information
    """
    with open(gt_file, encoding='latin1') as f:
        anno_info = []
        for line in f:
            line = line.strip('\n')
            if line[0] == '[' or line[0] == 'x':
                continue
            ann = line.split(',')
            bbox = ann[0:4]
            bbox = [int(coord) for coord in bbox]
            x, y, w, h = bbox
            segmentation = [x, y, x + w, y, x + w, y + h, x, y + h]
            anno = dict(
                iscrowd=0,
                category_id=1,
                bbox=bbox,
                area=w * h,
                segmentation=[segmentation])
            anno_info.append(anno)

    img_info.update(anno_info=anno_info)

    return img_info


def split_train_val_list(full_list, val_ratio):
    """Split list by val_ratio.

    Args:
        full_list (list): list to be split
        val_ratio (float): split ratio for val set

    return:
        list(list, list): train_list and val_list
    """
    n_total = len(full_list)
    offset = int(n_total * val_ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    val_list = full_list[:offset]
    train_list = full_list[offset:]
    return [train_list, val_list]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and val set of BID ')
    parser.add_argument('root_path', help='Root dir path of BID')
    parser.add_argument(
        '--nproc', default=1, type=int, help='Number of processes')
    parser.add_argument(
        '--val-ratio', help='Split ratio for val set', default=0., type=float)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path
    with mmcv.Timer(print_tmpl='It takes {}s to convert BID annotation'):
        files = collect_files(
            osp.join(root_path, 'imgs'), osp.join(root_path, 'annotations'))
        image_infos = collect_annotations(files, nproc=args.nproc)
        if args.val_ratio:
            image_infos = split_train_val_list(image_infos, args.val_ratio)
            splits = ['training', 'val']
        else:
            image_infos = [image_infos]
            splits = ['training']
        for i, split in enumerate(splits):
            convert_annotations(
                image_infos[i],
                osp.join(root_path, 'instances_' + split + '.json'))


if __name__ == '__main__':
    main()
