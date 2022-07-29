# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os.path as osp

import cv2

from mmocr.utils import list_from_file, list_to_file


def parse_old_label(data_root, in_path, img_size=False):
    imgid2imgname = {}
    imgid2anno = {}
    idx = 0
    for line in list_from_file(in_path):
        line = line.strip().split()
        img_full_path = osp.join(data_root, line[0])
        if not osp.exists(img_full_path):
            continue
        ann_file = osp.join(data_root, line[1])
        if not osp.exists(ann_file):
            continue

        img_info = {}
        img_info['file_name'] = line[0]
        if img_size:
            img = cv2.imread(img_full_path)
            h, w = img.shape[:2]
            img_info['height'] = h
            img_info['width'] = w
        imgid2imgname[idx] = img_info

        imgid2anno[idx] = []
        char_annos = []
        for t, ann_line in enumerate(list_from_file(ann_file)):
            ann_line = ann_line.strip()
            if t == 0:
                img_info['text'] = ann_line
            else:
                char_box = [float(x) for x in ann_line.split()]
                char_text = img_info['text'][t - 1]
                char_ann = dict(char_box=char_box, char_text=char_text)
                char_annos.append(char_ann)
        imgid2anno[idx] = char_annos
        idx += 1

    return imgid2imgname, imgid2anno


def gen_line_dict_file(out_path, imgid2imgname, imgid2anno, img_size=False):
    lines = []
    for key, value in imgid2imgname.items():
        if key in imgid2anno:
            anno = imgid2anno[key]
            line_dict = {}
            line_dict['file_name'] = value['file_name']
            line_dict['text'] = value['text']
            if img_size:
                line_dict['height'] = value['height']
                line_dict['width'] = value['width']
            line_dict['annotations'] = anno
            lines.append(json.dumps(line_dict))
    list_to_file(out_path, lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-root', help='data root for both image file and anno file')
    parser.add_argument(
        '--in-path',
        help='mapping file of image_name and ann_file,'
        ' "image_name ann_file" in each line')
    parser.add_argument(
        '--out-path', help='output txt path with line-json format')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    imgid2imgname, imgid2anno = parse_old_label(args.data_root, args.in_path)
    gen_line_dict_file(args.out_path, imgid2imgname, imgid2anno)
    print('finish')


if __name__ == '__main__':
    main()
