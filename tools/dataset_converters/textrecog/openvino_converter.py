# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import os.path as osp
from argparse import ArgumentParser
from functools import partial

import mmcv
from PIL import Image

from mmocr.utils import dump_ocr_data


def parse_args():
    parser = ArgumentParser(description='Generate training and validation set '
                            'of OpenVINO annotations for Open '
                            'Images by cropping box image.')
    parser.add_argument(
        'root_path', help='Root dir containing images and annotations')
    parser.add_argument(
        'n_proc', default=1, type=int, help='Number of processes to run')
    args = parser.parse_args()
    return args


def process_img(args, src_image_root, dst_image_root):
    # Dirty hack for multi-processing
    img_idx, img_info, anns = args
    src_img = Image.open(osp.join(src_image_root, img_info['file_name']))
    labels = []
    for ann_idx, ann in enumerate(anns):
        attrs = ann['attributes']
        text_label = attrs['transcription']

        # Ignore illegible or non-English words
        if not attrs['legible'] or attrs['language'] != 'english':
            continue

        x, y, w, h = ann['bbox']
        x, y = max(0, math.floor(x)), max(0, math.floor(y))
        w, h = math.ceil(w), math.ceil(h)
        dst_img = src_img.crop((x, y, x + w, y + h))
        dst_img_name = f'img_{img_idx}_{ann_idx}.jpg'
        dst_img_path = osp.join(dst_image_root, dst_img_name)
        # Preserve JPEG quality
        dst_img.save(dst_img_path, qtables=src_img.quantization)
        labels.append({
            'file_name': dst_img_name,
            'anno_info': [{
                'text': text_label
            }]
        })
    src_img.close()
    return labels


def convert_openimages(root_path,
                       dst_image_path,
                       dst_label_filename,
                       annotation_filename,
                       img_start_idx=0,
                       nproc=1):
    annotation_path = osp.join(root_path, annotation_filename)
    if not osp.exists(annotation_path):
        raise Exception(
            f'{annotation_path} not exists, please check and try again.')
    src_image_root = root_path

    # outputs
    dst_label_file = osp.join(root_path, dst_label_filename)
    dst_image_root = osp.join(root_path, dst_image_path)
    os.makedirs(dst_image_root, exist_ok=True)

    annotation = mmcv.load(annotation_path)

    process_img_with_path = partial(
        process_img,
        src_image_root=src_image_root,
        dst_image_root=dst_image_root)
    tasks = []
    anns = {}
    for ann in annotation['annotations']:
        anns.setdefault(ann['image_id'], []).append(ann)
    for img_idx, img_info in enumerate(annotation['images']):
        tasks.append((img_idx + img_start_idx, img_info, anns[img_info['id']]))
    labels_list = mmcv.track_parallel_progress(
        process_img_with_path, tasks, keep_order=True, nproc=nproc)
    final_labels = []
    for label_list in labels_list:
        final_labels += label_list
    dump_ocr_data(final_labels, dst_label_file, 'textrecog')
    return len(annotation['images'])


def main():
    args = parse_args()
    root_path = args.root_path
    print('Processing training set...')
    num_train_imgs = 0
    for s in '125f':
        num_train_imgs = convert_openimages(
            root_path=root_path,
            dst_image_path=f'image_{s}',
            dst_label_filename=f'train_{s}_label.json',
            annotation_filename=f'text_spotting_openimages_v5_train_{s}.json',
            img_start_idx=num_train_imgs,
            nproc=args.n_proc)
    print('Processing validation set...')
    convert_openimages(
        root_path=root_path,
        dst_image_path='image_val',
        dst_label_filename='val_label.json',
        annotation_filename='text_spotting_openimages_v5_validation.json',
        img_start_idx=num_train_imgs,
        nproc=args.n_proc)
    print('Finish')


if __name__ == '__main__':
    main()
