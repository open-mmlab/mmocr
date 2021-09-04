# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os.path as osp
import xml.etree.ElementTree as ET
from functools import partial

import mmcv
import numpy as np
from shapely.geometry import Polygon

from mmocr.utils import convert_annotations, list_from_file


def collect_files(img_dir, gt_dir, split):
    """Collect all images and their corresponding groundtruth files.

    Args:
        img_dir(str): The image directory
        gt_dir(str): The groundtruth directory
        split(str): The split of dataset. Namely: training or test

    Returns:
        files(list): The list of tuples (img_file, groundtruth_file)
    """
    assert isinstance(img_dir, str)
    assert img_dir
    assert isinstance(gt_dir, str)
    assert gt_dir

    # note that we handle png and jpg only. Pls convert others such as gif to
    # jpg or png offline
    suffixes = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']

    imgs_list = []
    for suffix in suffixes:
        imgs_list.extend(glob.glob(osp.join(img_dir, '*' + suffix)))

    files = []
    if split == 'training':
        for img_file in imgs_list:
            gt_file = gt_dir + '/' + osp.splitext(
                osp.basename(img_file))[0] + '.xml'
            files.append((img_file, gt_file))
        assert len(files), f'No images found in {img_dir}'
        print(f'Loaded {len(files)} images from {img_dir}')
    elif split == 'test':
        for img_file in imgs_list:
            gt_file = gt_dir + '/000' + osp.splitext(
                osp.basename(img_file))[0] + '.txt'
            files.append((img_file, gt_file))
        assert len(files), f'No images found in {img_dir}'
        print(f'Loaded {len(files)} images from {img_dir}')

    return files


def collect_annotations(files, split, nproc=1):
    """Collect the annotation information.

    Args:
        files(list): The list of tuples (image_file, groundtruth_file)
        split(str): The split of dataset. Namely: training or test
        nproc(int): The number of process to collect annotations

    Returns:
        images(list): The list of image information dicts
    """
    assert isinstance(files, list)
    assert isinstance(split, str)
    assert isinstance(nproc, int)

    load_img_info_with_split = partial(load_img_info, split=split)
    if nproc > 1:
        images = mmcv.track_parallel_progress(
            load_img_info_with_split, files, nproc=nproc)
    else:
        images = mmcv.track_progress(load_img_info_with_split, files)

    return images


def load_txt_info(gt_file, img_info):
    anno_info = []
    for line in list_from_file(gt_file):
        # each line has one ploygen (n vetices), and one text.
        # e.g., 695,885,866,888,867,1146,696,1143,####Latin 9
        line = line.strip()
        strs = line.split(',')
        category_id = 1
        assert strs[28][0] == '#'
        xy = [int(x) for x in strs[0:28]]
        assert len(xy) == 28
        coordinates = np.array(xy).reshape(-1, 2)
        polygon = Polygon(coordinates)
        iscrowd = 0
        area = polygon.area
        # convert to COCO style XYWH format
        min_x, min_y, max_x, max_y = polygon.bounds
        bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

        anno = dict(
            iscrowd=iscrowd,
            category_id=category_id,
            bbox=bbox,
            area=area,
            segmentation=[xy])
        anno_info.append(anno)
    img_info.update(anno_info=anno_info)
    return img_info


def load_xml_info(gt_file, img_info):

    obj = ET.parse(gt_file)
    anno_info = []
    for image in obj.getroot():  # image
        for box in image:  # image
            h = box.attrib['height']
            w = box.attrib['width']
            x = box.attrib['left']
            y = box.attrib['top']
            # label = box[0].text
            segs = box[1].text
            pts = segs.strip().split(',')
            pts = [int(x) for x in pts]
            assert len(pts) == 28
            # pts = []
            # for iter in range(2,len(box)):
            #    pts.extend([int(box[iter].attrib['x']),
            #  int(box[iter].attrib['y'])])
            iscrowd = 0
            category_id = 1
            bbox = [int(x), int(y), int(w), int(h)]

            coordinates = np.array(pts).reshape(-1, 2)
            polygon = Polygon(coordinates)
            area = polygon.area
            anno = dict(
                iscrowd=iscrowd,
                category_id=category_id,
                bbox=bbox,
                area=area,
                segmentation=[pts])
            anno_info.append(anno)

    img_info.update(anno_info=anno_info)

    return img_info


def load_img_info(files, split):
    """Load the information of one image.

    Args:
        files(tuple): The tuple of (img_file, groundtruth_file)
        split(str): The split of dataset: training or test

    Returns:
        img_info(dict): The dict of the img and annotation information
    """
    assert isinstance(files, tuple)
    assert isinstance(split, str)

    img_file, gt_file = files
    # read imgs with ignoring orientations
    img = mmcv.imread(img_file, 'unchanged')

    split_name = osp.basename(osp.dirname(img_file))
    img_info = dict(
        # remove img_prefix for filename
        file_name=osp.join(split_name, osp.basename(img_file)),
        height=img.shape[0],
        width=img.shape[1],
        # anno_info=anno_info,
        segm_file=osp.join(split_name, osp.basename(gt_file)))

    if split == 'training':
        img_info = load_xml_info(gt_file, img_info)
    elif split == 'test':
        img_info = load_txt_info(gt_file, img_info)
    else:
        raise NotImplementedError

    return img_info


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert ctw1500 annotations to COCO format')
    parser.add_argument('root_path', help='ctw1500 root path')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--split-list',
        nargs='+',
        help='a list of splits. e.g., "--split-list training test"')

    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path
    out_dir = args.out_dir if args.out_dir else root_path
    mmcv.mkdir_or_exist(out_dir)

    img_dir = osp.join(root_path, 'imgs')
    gt_dir = osp.join(root_path, 'annotations')

    set_name = {}
    for split in args.split_list:
        set_name.update({split: 'instances_' + split + '.json'})
        assert osp.exists(osp.join(img_dir, split))

    for split, json_name in set_name.items():
        print(f'Converting {split} into {json_name}')
        with mmcv.Timer(print_tmpl='It takes {}s to convert icdar annotation'):
            files = collect_files(
                osp.join(img_dir, split), osp.join(gt_dir, split), split)
            image_infos = collect_annotations(files, split, nproc=args.nproc)
            convert_annotations(image_infos, osp.join(out_dir, json_name))


if __name__ == '__main__':
    main()
