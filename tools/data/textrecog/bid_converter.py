# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
import os.path as osp

import mmcv

from mmocr.datasets.pipelines.crop import crop_img
from mmocr.utils.fileio import list_to_file


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
    Args:
        gt_file (str): The path to ground-truth
        img_info (dict): The dict of the img and annotation information
    Returns:
        img_info (dict): The dict of the img and annotation information
    """

    with open(gt_file, 'r', encoding='latin1') as f:
        anno_info = []
        for line in f:
            line = line.strip('\n')
            # Ignore hard samples
            if line[0] == '[' or line[0] == 'x':
                continue
            ann = line.split(',')
            bbox = ann[0:4]
            bbox = [int(_) for _ in bbox]
            x, y, w, h = bbox
            # in case ',' exists in label
            word = ','.join(ann[4:]) if len(ann[4:]) > 1 else ann[4]
            # remove the initial space
            word = word.strip()
            bbox = [x, y, x + w, y, x + w, y + h, x, y + h]

            anno = dict(bbox=bbox, word=word)
            anno_info.append(anno)

    img_info.update(anno_info=anno_info)

    return img_info


def split_train_test_list(full_list, test_ratio):
    """Split list by test_ratio
    Args:
        full_list (list): List to be split
        test_ratio (float): Ratio for test set
    return:
        list(list, list): train_list and test_list
    """

    n_total = len(full_list)
    offset = int(n_total * test_ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    test_list = full_list[:offset]
    train_list = full_list[offset:]
    return [train_list, test_list]


def generate_ann(root_path, image_infos, preserve_vertical, test_ratio,
                 format):
    """Generate cropped annotations and label txt file.
    Args:
        root_path (str): The root path of the dataset
        split (str): The split of dataset. Namely: training or test
        image_infos (list[dict]): A list of dicts of the img and
            annotation information
        preserve_vertical (bool): Whether to preserve vertical texts
        test_ratio (float): Ratio of test set from the whole dataset
        format (str): Using jsonl(dict) or str to format annotations
    """

    assert test_ratio <= 1.

    if test_ratio:
        image_infos = split_train_test_list(image_infos, test_ratio)
        splits = ['training', 'test']

    else:
        image_infos = [image_infos]
        splits = ['training']

    for i, split in enumerate(splits):
        dst_image_root = osp.join(root_path, 'dst_imgs', split)
        dst_label_file = osp.join(root_path, f'{split}_label.txt')
        os.makedirs(dst_image_root, exist_ok=True)

        lines = []
        for image_info in image_infos[i]:
            index = 1
            src_img_path = osp.join(root_path, 'imgs', image_info['file_name'])
            image = mmcv.imread(src_img_path)
            src_img_root = image_info['file_name'].split('.')[0]

            for anno in image_info['anno_info']:
                word = anno['word']
                dst_img = crop_img(image, anno['bbox'])
                h, w, _ = dst_img.shape

                # Skip invalid annotations
                if min(dst_img.shape) == 0:
                    continue
                # Skip vertical texts
                if not preserve_vertical and h / w > 2:
                    continue

                dst_img_name = f'{src_img_root}_{index}.png'
                index += 1
                dst_img_path = osp.join(dst_image_root, dst_img_name)
                mmcv.imwrite(dst_img, dst_img_path)
                if format == 'txt':
                    lines.append(
                        f'{osp.basename(dst_image_root)}/{dst_img_name} '
                        f'{word}')
                elif format == 'jsonl':
                    lines.append(
                        json.dumps(
                            {
                                'filename': f'{osp.basename(dst_image_root)} \
                                  /{dst_img_name}',
                                'text': word
                            },
                            ensure_ascii=False))
                else:
                    raise NotImplementedError
    list_to_file(dst_label_file, lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and test set of BID ')
    parser.add_argument('root_path', help='Root dir path of BID')
    parser.add_argument(
        '--preserve_vertical',
        help='Preserve samples containing vertical texts',
        action='store_true')
    parser.add_argument(
        '--test_ratio',
        help='Ratio of test set from the whole dataset',
        default=0.)
    parser.add_argument(
        '--nproc', default=1, type=int, help='Number of processes')
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
    with mmcv.Timer(print_tmpl='It takes {}s to convert BID annotation'):
        files = collect_files(
            osp.join(root_path, 'imgs'), osp.join(root_path, 'annotations'))
        image_infos = collect_annotations(files, nproc=args.nproc)
        generate_ann(root_path, image_infos, args.preserve_vertical,
                     float(args.test_ratio), args.format)


if __name__ == '__main__':
    main()
