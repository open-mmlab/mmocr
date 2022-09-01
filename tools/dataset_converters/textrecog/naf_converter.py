# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import mmcv
import mmengine
import numpy as np

from mmocr.utils import crop_img, dump_ocr_data


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
        'filedBBs': [{
            'poly_points': [[435,1406], [466,1406], [466,1439], [435,1439]],
            "type": "fieldCheckBox",
            "id": "f0",
            "isBlank": 1, # 0:text,1:handwriting,2:print,3:blank,4:signature,
        }], ...
        "transcriptions":{
            "f38": "CASE NUMBER",
            "f29": "July 1, 1949",
            "t20": "RANK",
            "t19": "COMPANY",
            ...
        }
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


def generate_ann(root_path, split, image_infos, preserve_vertical):
    """Generate cropped annotations and label txt file.

    Args:
        root_path (str): The root path of the dataset
        split (str): The split of dataset. Namely: training or test
        image_infos (list[dict]): A list of dicts of the img and
            annotation information
        preserve_vertical (bool): Whether to preserve vertical texts
    """

    dst_image_root = osp.join(root_path, 'crops', split)
    ignore_image_root = osp.join(root_path, 'ignores', split)
    if split == 'training':
        dst_label_file = osp.join(root_path, 'train_label.json')
    elif split == 'val':
        dst_label_file = osp.join(root_path, 'val_label.json')
    elif split == 'test':
        dst_label_file = osp.join(root_path, 'test_label.json')
    else:
        raise NotImplementedError
    mmengine.mkdir_or_exist(dst_image_root)
    mmengine.mkdir_or_exist(ignore_image_root)

    img_info = []
    for image_info in image_infos:
        index = 1
        src_img_path = osp.join(root_path, 'imgs', image_info['file_name'])
        image = mmcv.imread(src_img_path)
        src_img_root = image_info['file_name'].split('.')[0]

        for anno in image_info['anno_info']:
            word = anno['word']
            word = word.strip('\u202a')  # Remove unicode control character
            word = word.replace('»',
                                '').replace('«',
                                            '')  # Remove strikethrough flag
            dst_img = crop_img(image, anno['bbox'], 0, 0)
            h, w, _ = dst_img.shape

            dst_img_name = f'{src_img_root}_{index}.png'
            index += 1
            # Skip invalid and illegible annotations
            if min(dst_img.shape) == 0 or '§' in word or '¿' in word or len(
                    word) == 0:
                continue
            # Skip vertical texts
            # (Do Not Filter For Val and Test Split)
            if (not preserve_vertical and h / w > 2) and split == 'training':
                dst_img_path = osp.join(ignore_image_root, dst_img_name)
                mmcv.imwrite(dst_img, dst_img_path)
                continue

            dst_img_path = osp.join(dst_image_root, dst_img_name)
            mmcv.imwrite(dst_img, dst_img_path)

            img_info.append({
                'file_name': dst_img_name,
                'anno_info': [{
                    'text': word
                }]
            })

    dump_ocr_data(img_info, dst_label_file, 'textrecog')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training, val, and test set of NAF ')
    parser.add_argument('root_path', help='Root dir path of NAF')
    parser.add_argument(
        '--preserve-vertical',
        help='Preserve samples containing vertical texts',
        action='store_true')
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
        with mmengine.Timer(
                print_tmpl='It takes {}s to convert NAF annotation'):
            files = collect_files(
                osp.join(root_path, 'imgs'),
                osp.join(root_path, 'annotations'), split_info[split])
            image_infos = collect_annotations(files, nproc=args.nproc)
            generate_ann(root_path, split, image_infos, args.preserve_vertical)


if __name__ == '__main__':
    main()
