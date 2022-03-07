# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import mmcv
import numpy as np

from mmocr.datasets.pipelines.crop import crop_img
from mmocr.utils.fileio import list_to_file


def collect_files(img_dir, gt_dir, split_info):
    """Collect all images and their corresponding groundtruth files.

    Args:
        img_dir (str): The image directory
        gt_dir (str): The groundtruth directory
        split_info (dict): The split information for train/val/test

    Returns:
        files (list): The list of tuples (img_file, groundtruth_file)
    """
    assert isinstance(img_dir, str)
    assert img_dir
    assert isinstance(gt_dir, str)
    assert gt_dir
    assert isinstance(split_info, dict)
    assert split_info

    ann_list, imgs_list = [], []
    for group in split_info:
        for img in split_info[group]:
            image_path = osp.join(img_dir, img)
            anno_path = osp.join(gt_dir, 'groups', group,
                                 img.replace('jpg', 'json'))

            # Filtering out the missing images
            if not osp.exists(image_path) or not osp.exists(anno_path):
                continue

            imgs_list.append(image_path)
            ann_list.append(anno_path)

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

    Args:
        gt_file (str): The path to ground-truth
        img_info (dict): The dict of the img and annotation information

    Returns:
        img_info (dict): The dict of the img and annotation information
    """
    assert isinstance(gt_file, str)
    assert isinstance(img_info, dict)

    annotation = mmcv.load(gt_file)
    anno_info = []

    # 'textBBs' contains the printed texts of the table while 'fieldBBs'
    #  contains the text filled by human.
    for box_type in ['textBBs', 'fieldBBs']:
        # NAF dataset only provides transcription GT for 'filedBBs', the
        # 'textBBs' is only used for detection task.
        if box_type == 'textBBs':
            continue
        for anno in annotation[box_type]:
            # Skip images containing detection annotations only
            if 'transcriptions' not in annotation.keys():
                continue
            # Skip boxes without recognition GT
            if anno['id'] not in annotation['transcriptions'].keys():
                continue

            word = annotation['transcriptions'][anno['id']]
            # Skip blank boxes
            if len(word) == 0:
                continue

            bbox = np.array(anno['poly_points']).reshape(1, 8)[0].tolist()

            anno = dict(bbox=bbox, word=word)
            anno_info.append(anno)

    img_info.update(anno_info=anno_info)

    return img_info


def generate_ann(root_path, split, image_infos, preserve_vertical,
                 preserve_special_symbols):
    """Generate cropped annotations and label txt file.

    Args:
        root_path (str): The root path of the dataset
        split (str): The split of dataset. Namely: training or test
        image_infos (list[dict]): A list of dicts of the img and
            annotation information
        preserve_vertical (bool): Whether to preserve vertical texts
    """

    dst_image_root = osp.join(root_path, 'dst_imgs', split)
    if split == 'training':
        dst_label_file = osp.join(root_path, 'train_label.txt')
    elif split == 'val':
        dst_label_file = osp.join(root_path, 'val_label.txt')
    elif split == 'test':
        dst_label_file = osp.join(root_path, 'test_label.txt')
    else:
        raise NotImplementedError
    mmcv.mkdir_or_exist(dst_image_root)

    lines = []
    for image_info in image_infos:
        index = 1
        src_img_path = osp.join(root_path, 'imgs', image_info['file_name'])
        image = mmcv.imread(src_img_path)
        src_img_root = image_info['file_name'].split('.')[0]

        for anno in image_info['anno_info']:
            word = anno['word']
            word = word.strip('\u202a')  # remove unicode control character
            dst_img = crop_img(image, anno['bbox'])
            h, w, _ = dst_img.shape

            # Skip invalid annotations
            if min(dst_img.shape) == 0:
                continue
            # Skip vertical texts
            if not preserve_vertical and h / w > 2:
                continue
            # Skip non-ASCII characters
            if not preserve_special_symbols and len(word) != len(
                    word.encode()):
                continue

            dst_img_name = f'{src_img_root}_{index}.png'
            index += 1
            dst_img_path = osp.join(dst_image_root, dst_img_name)
            mmcv.imwrite(dst_img, dst_img_path)
            lines.append(f'{osp.basename(dst_image_root)}/{dst_img_name} '
                         f'{word}')
    list_to_file(dst_label_file, lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training, val, and test set of NAF ')
    parser.add_argument('root_path', help='Root dir path of NAF')
    parser.add_argument(
        '--preserve-vertical',
        help='Preserve samples containing vertical texts',
        action='store_true')
    parser.add_argument(
        '--preserve-special-symbols',
        help='Preserve non-ASCII characters such as tick and section sign',
        action='store_true')
    parser.add_argument(
        '--nproc', default=1, type=int, help='Number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path
    split_info = mmcv.load(
        osp.join(root_path, 'annotations', 'train_valid_test_split.json'))
    split_info['training'] = split_info.pop('train')
    split_info['val'] = split_info.pop('valid')
    for split in ['training', 'val', 'test']:
        print(f'Processing {split} set...')
        with mmcv.Timer(print_tmpl='It takes {}s to convert NAF annotation'):
            files = collect_files(
                osp.join(root_path, 'imgs'),
                osp.join(root_path, 'annotations'), split_info[split])
            image_infos = collect_annotations(files, nproc=args.nproc)
            generate_ann(root_path, split, image_infos, args.preserve_vertical,
                         args.preserve_special_symbols)


if __name__ == '__main__':
    main()
