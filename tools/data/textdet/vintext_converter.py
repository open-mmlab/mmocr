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
        ann_file = 'gt_' + str(int(img_file[2:6])) + '.txt'
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
    assert int(osp.basename(gt_file)[3:-4]) == int(
        osp.basename(img_file)[2:-4])
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

    Args:
        gt_file (str): The path to ground-truth
        img_info (dict): The dict of the img and annotation information

    Returns:
        img_info (dict): The dict of the img and annotation information
    """

    with open(gt_file, 'r', encoding='utf-8') as f:
        anno_info = []
        for line in f:
            line = line.strip('\n')
            ann = line.split(',')
            bbox = ann[0:8]
            word = line[len(','.join(bbox)) + 1:]
            bbox = [int(_) for _ in bbox]
            segmentation = bbox
            x_min = min(bbox[0], bbox[2], bbox[4], bbox[6])
            x_max = max(bbox[0], bbox[2], bbox[4], bbox[6])
            y_min = min(bbox[1], bbox[3], bbox[5], bbox[7])
            y_max = min(bbox[1], bbox[3], bbox[5], bbox[7])
            w = x_max - x_min
            h = y_max - y_min
            bbox = [x_min, y_min, w, h]
            iscrowd = 1 if word == '###' else 0

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
        description='Generate training and test set of VinText ')
    parser.add_argument('root_path', help='Root dir path of VinText')
    parser.add_argument(
        '--nproc', default=1, type=int, help='Number of processes')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path
    for split in ['training', 'test', 'unseen_test']:
        print(f'Processing {split} set...')
        with mmcv.Timer(
                print_tmpl='It takes {}s to convert VinText annotation'):
            files = collect_files(
                osp.join(root_path, 'imgs', split),
                osp.join(root_path, 'annotations'))
            image_infos = collect_annotations(files, nproc=args.nproc)
            convert_annotations(
                image_infos, osp.join(root_path,
                                      'instances_' + split + '.json'))


if __name__ == '__main__':
    main()
