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
        imgs_list.append(osp.join(img_dir, ann_file.replace('json', 'jpg')))

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

    if osp.splitext(gt_file)[1] == '.json':
        img_info = load_json_info(gt_file, img_info)
    else:
        raise NotImplementedError

    return img_info


def load_json_info(gt_file, img_info):
    """Collect the annotation information.

    The annotation format is as the following:

    {
        "chars": [
            {
                "ignore": 0,
                "transcription": "H",
                "points": [25, 175, 112, 175, 112, 286, 25, 286]
            },
            {
                "ignore": 0,
                "transcription": "O",
                "points": [102, 182, 210, 182, 210, 273, 102, 273]
            }, ...
        ]
        "lines": [
            {
                "ignore": 0,
                "transcription": "HOKI",
                "points": [23, 173, 327, 180, 327, 290, 23, 283]
            },
            {
                "ignore": 0,
                "transcription": "TEA",
                "points": [368, 180, 621, 180, 621, 294, 368, 294]
            }, ...
        ]
    }


    Args:
        gt_file (str): The path to ground-truth
        img_info (dict): The dict of the img and annotation information

    Returns:
        img_info (dict): The dict of the img and annotation information
    """

    annotation = mmcv.load(gt_file)
    anno_info = []
    for line in annotation['lines']:
        segmentation = line['points']
        x = max(0, min(segmentation[0::2]))
        y = max(0, min(segmentation[1::2]))
        w = abs(max(segmentation[0::2]) - x)
        h = abs(max(segmentation[1::2]) - y)
        bbox = [x, y, w, h]

        anno = dict(
            iscrowd=line['ignore'],
            category_id=1,
            bbox=bbox,
            area=w * h,
            segmentation=[segmentation])
        anno_info.append(anno)

    img_info.update(anno_info=anno_info)

    return img_info


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and val set of ReCTS.')
    parser.add_argument('root_path', help='Root dir path of ReCTS')
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
    trn_infos = collect_annotations(trn_files, nproc=args.nproc)
    with mmcv.Timer(
            print_tmpl='It takes {}s to convert ReCTS Training annotation'):
        convert_annotations(trn_infos,
                            osp.join(root_path, 'instances_training.json'))

    # Val set
    if len(val_files) > 0:
        val_infos = collect_annotations(val_files, nproc=args.nproc)
        with mmcv.Timer(
                print_tmpl='It takes {}s to convert ReCTS Val annotation'):
            convert_annotations(val_infos,
                                osp.join(root_path, 'instances_val.json'))


if __name__ == '__main__':
    main()
