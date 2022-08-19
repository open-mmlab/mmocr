# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import mmcv
import mmengine

from mmocr.utils import dump_ocr_data


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
    # Read imgs while ignoring orientations
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

    Annotation Format
    {
        'textBBs': [{
            'poly_points': [[435,1406], [466,1406], [466,1439], [435,1439]],
            "type": "text",
            "id": "t1",
        }], ...
    }

    Some special characters are used in the transcription:
    "«text»" indicates that "text" had a strikethrough
    "¿" indicates the transcriber could not read a character
    "§" indicates the whole line or word was illegible
    "" (empty string) is if the field was blank

    Args:
        gt_file (str): The path to ground-truth
        img_info (dict): The dict of the img and annotation information

    Returns:
        img_info (dict): The dict of the img and annotation information
    """
    assert isinstance(gt_file, str)
    assert isinstance(img_info, dict)

    annotation = mmengine.load(gt_file)
    anno_info = []

    # 'textBBs' contains the printed texts of the table while 'fieldBBs'
    #  contains the text filled by human.
    for box_type in ['textBBs', 'fieldBBs']:
        for anno in annotation[box_type]:
            # Skip blanks
            if box_type == 'fieldBBs':
                if anno['type'] == 'blank':
                    continue

            xs, ys, segmentation = [], [], []
            for p in anno['poly_points']:
                xs.append(p[0])
                ys.append(p[1])
                segmentation.append(p[0])
                segmentation.append(p[1])
            x, y = max(0, min(xs)), max(0, min(ys))
            w, h = max(xs) - x, max(ys) - y
            bbox = [x, y, w, h]

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
        description='Generate training, val, and test set of NAF ')
    parser.add_argument('root_path', help='Root dir path of NAF')
    parser.add_argument(
        '--nproc', default=1, type=int, help='Number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path
    split_info = mmengine.load(
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
            dump_ocr_data(image_infos,
                          osp.join(root_path, 'instances_' + split + '.json'),
                          'textdet')


if __name__ == '__main__':
    main()
