# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import xml.etree.ElementTree as ET

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
        ann_path = osp.join(gt_dir, img_file.split('.')[0] + '.xml')
        if os.path.exists(ann_path):
            ann_list.append(ann_path)
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
    assert osp.basename(gt_file).split('.')[0] == osp.basename(img_file).split(
        '.')[0]
    # read imgs while ignoring orientations
    img = mmcv.imread(img_file, 'unchanged')

    try:
        img_info = dict(
            file_name=osp.join(osp.basename(img_file)),
            height=img.shape[0],
            width=img.shape[1],
            segm_file=osp.join(osp.basename(gt_file)))
    except AttributeError:
        print(f'Skip broken img {img_file}')
        return None

    if osp.splitext(gt_file)[1] == '.xml':
        img_info = load_xml_info(gt_file, img_info)
    else:
        raise NotImplementedError

    return img_info


def load_xml_info(gt_file, img_info):
    """Collect the annotation information.

    The annotation format is as the following:
    <annotations>
    ...
        <object>
            <name>SMT</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>157</xmin>
                <ymin>294</ymin>
                <xmax>237</xmax>
                <ymax>357</ymax>
            </bndbox>
        <object>

    Args:
        gt_file (str): The path to ground-truth
        img_info (dict): The dict of the img and annotation information

    Returns:
        img_info (dict): The dict of the img and annotation information
    """
    obj = ET.parse(gt_file)
    root = obj.getroot()
    anno_info = []
    for object in root.iter('object'):
        word = object.find('name').text
        iscrowd = 1 if len(word) == 0 else 0
        x1 = int(object.find('bndbox').find('xmin').text)
        y1 = int(object.find('bndbox').find('ymin').text)
        x2 = int(object.find('bndbox').find('xmax').text)
        y2 = int(object.find('bndbox').find('ymax').text)

        x = max(0, min(x1, x2))
        y = max(0, min(y1, y2))
        w, h = abs(x2 - x1), abs(y2 - y1)
        bbox = [x1, y1, w, h]
        segmentation = [x, y, x + w, y, x + w, y + h, x, y + h]
        anno = dict(
            iscrowd=iscrowd,
            category_id=1,
            bbox=bbox,
            area=w * h,
            segmentation=[segmentation])
        anno_info.append(anno)

    img_info.update(anno_info=anno_info)

    return img_info


def split_train_val_list(full_list, val_ratio):
    """Split list by val_ratio

    Args:
        full_list (list): List to be splited
        val_ratio (float): Split ratio for val set

    return:
        list(list, list): Train_list and val_list
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
        description='Generate training and val set of ILST ')
    parser.add_argument('root_path', help='Root dir path of ILST')
    parser.add_argument(
        '--val-ratio', help='Split ratio for val set', default=0.2, type=float)
    parser.add_argument(
        '--nproc', default=1, type=int, help='Number of processes')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path
    with mmcv.Timer(print_tmpl='It takes {}s to convert ILST annotation'):
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
                list(filter(None, image_infos[i])),
                osp.join(root_path, 'instances_' + split + '.json'))


if __name__ == '__main__':
    main()
