# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import mmcv
from PIL import Image

import mmengine
from mmocr.utils import dump_ocr_data


def convert_gif(img_path):
    """Convert the gif image to png format.

    Args:
        img_path (str): The path to the gif image
    """
    img = Image.open(img_path)
    dst_path = img_path.replace('gif', 'png')
    img.save(dst_path)
    os.remove(img_path)
    print(f'Convert {img_path} to {dst_path}')


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
    for img in os.listdir(img_dir):
        img_path = osp.join(img_dir, img)
        # mmcv cannot read gif images, so convert them to png
        if img.endswith('gif'):
            convert_gif(img_path)
            img_path = img_path.replace('gif', 'png')
        imgs_list.append(img_path)
        ann_list.append(osp.join(gt_dir, 'gt_' + img.split('.')[0] + '.txt'))

    files = list(zip(sorted(imgs_list), sorted(ann_list)))
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
        images = mmengine.track_parallel_progress(
            load_img_info, files, nproc=nproc)
    else:
        images = mmengine.track_progress(load_img_info, files)

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
    # read imgs while ignoring orientations
    img = mmcv.imread(img_file, 'unchanged')

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
    left, top, right, bottom, "transcription"

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
            xmin, ymin, xmax, ymax = line.split(',')[0:4]
            x = max(0, int(xmin))
            y = max(0, int(ymin))
            w = int(xmax) - x
            h = int(ymax) - y
            bbox = [x, y, w, h]
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and test set of IC11')
    parser.add_argument('root_path', help='Root dir path of IC11')
    parser.add_argument(
        '--nproc', default=1, type=int, help='Number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path

    for split in ['training', 'test']:
        print(f'Processing {split} set...')
        with mmengine.Timer(print_tmpl='It takes {}s to convert annotation'):
            files = collect_files(
                osp.join(root_path, 'imgs', split),
                osp.join(root_path, 'annotations', split))
            image_infos = collect_annotations(files, nproc=args.nproc)
            dump_ocr_data(image_infos,
                          osp.join(root_path, 'instances_' + split + '.json'),
                          'textdet')


if __name__ == '__main__':
    main()
