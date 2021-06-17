import argparse
import gc
import json
import math
import os
import os.path as osp

import cv2

from mmocr.utils.fileio import list_to_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and validation set of TextOCR '
        'by cropping box image.')
    parser.add_argument('root_path', help='Root dir path of TextOCR')
    args = parser.parse_args()
    return args


def convert_textocr(root_path,
                    dst_image_path,
                    dst_label_filename,
                    annotation_filename,
                    img_start_idx=0,
                    print_every=100):

    annotation_path = osp.join(root_path, annotation_filename)
    if not osp.exists(annotation_path):
        raise Exception(
            f'{annotation_path} not exists, please check and try again.')
    src_image_root = root_path

    # outputs
    dst_label_file = osp.join(root_path, dst_label_filename)
    dst_image_root = osp.join(root_path, dst_image_path)
    os.makedirs(dst_image_root, exist_ok=True)

    annotation = json.load(open(annotation_path, 'r'))
    labels = []

    img_idx = img_start_idx
    for img_info in annotation['imgs'].values():
        if img_idx > 0 and img_idx % print_every == 0:
            print(
                f'{img_idx-img_start_idx}/{len(annotation["imgs"].values())}')

        ann_ids = annotation['imgToAnns'][img_info['id']]
        src_img = cv2.imread(osp.join(src_image_root, img_info['file_name']))
        for ann_idx, ann_id in enumerate(ann_ids):
            ann = annotation['anns'][ann_id]
            text_label = ann['utf8_string'].lower()

            # Ignore illegible or non-English words
            if text_label == '.':
                continue

            x, y, w, h = ann['bbox']
            x, y = max(0, math.floor(x)), max(0, math.floor(y))
            w, h = math.ceil(w), math.ceil(h)
            dst_img = src_img[y:y + h, x:x + w]
            dst_img_name = f'img_{img_idx}_{ann_idx}.jpg'
            dst_img_path = osp.join(dst_image_root, dst_img_name)
            cv2.imwrite(dst_img_path, dst_img)
            del src_img
            del dst_img
            gc.collect()
            labels.append(f'{osp.basename(dst_image_root)}/{dst_img_name} '
                          f'{text_label}')
        img_idx += 1
    list_to_file(dst_label_file, labels)
    return img_idx


def main():
    args = parse_args()
    root_path = args.root_path
    print('Processing training set...')
    num_train_imgs = convert_textocr(
        root_path=root_path,
        dst_image_path='image',
        dst_label_filename='train_label.txt',
        annotation_filename='TextOCR_0.1_train.json',
        print_every=1000)
    print('Processing validation set...')
    convert_textocr(
        root_path=root_path,
        dst_image_path='image',
        dst_label_filename='val_label.txt',
        annotation_filename='TextOCR_0.1_val.json',
        img_start_idx=num_train_imgs,
        print_every=1000)
    print('Finish')


if __name__ == '__main__':
    main()
