# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math
import os
import os.path as osp

import mmcv
import mmengine
import numpy as np

from mmocr.utils import crop_img, dump_ocr_data


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training, validation and test set of IMGUR ')
    parser.add_argument('root_path', help='Root dir path of IMGUR')
    args = parser.parse_args()

    return args


def collect_imgur_info(root_path, annotation_filename, print_every=1000):

    annotation_path = osp.join(root_path, 'annotations', annotation_filename)
    if not osp.exists(annotation_path):
        raise Exception(
            f'{annotation_path} not exists, please check and try again.')

    annotation = mmengine.load(annotation_path)
    images = annotation['index_to_ann_map'].keys()
    img_infos = []
    for i, img_name in enumerate(images):
        if i >= 0 and i % print_every == 0:
            print(f'{i}/{len(images)}')

        img_path = osp.join(root_path, 'imgs', img_name + '.jpg')

        # Skip not exist images
        if not osp.exists(img_path):
            continue

        img = mmcv.imread(img_path, 'unchanged')

        # Skip broken images
        if img is None:
            continue

        img_info = dict(
            file_name=img_name + '.jpg',
            height=img.shape[0],
            width=img.shape[1])

        anno_info = []
        for ann_id in annotation['index_to_ann_map'][img_name]:
            ann = annotation['ann_id'][ann_id]

            # The original annotation is oriented rects [x, y, w, h, a]
            box = np.fromstring(
                ann['bounding_box'][1:-2], sep=',', dtype=float)
            bbox = convert_oriented_box(box)
            word = ann['word']

            anno = dict(bbox=bbox, word=word)
            anno_info.append(anno)
        img_info.update(anno_info=anno_info)
        img_infos.append(img_info)

    return img_infos


def convert_oriented_box(box):

    x_ctr, y_ctr, width, height, angle = box[:5]
    angle = -angle * math.pi / 180

    tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
    rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    poly = R.dot(rect)
    x0, x1, x2, x3 = poly[0, :4] + x_ctr
    y0, y1, y2, y3 = poly[1, :4] + y_ctr
    poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float32)
    poly = get_best_begin_point_single(poly)

    return poly.tolist()


def get_best_begin_point_single(coordinate):

    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combine = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
               [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
               [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
               [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combine[i][0], dst_coordinate[0]) \
            + cal_line_length(combine[i][1], dst_coordinate[1]) \
            + cal_line_length(combine[i][2], dst_coordinate[2]) \
            + cal_line_length(combine[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass

    return np.array(combine[force_flag]).reshape(8)


def cal_line_length(point1, point2):

    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) +
        math.pow(point1[1] - point2[1], 2))


def generate_ann(root_path, split, image_infos):

    dst_image_root = osp.join(root_path, 'crops', split)
    dst_label_file = osp.join(root_path, f'{split}_label.json')
    os.makedirs(dst_image_root, exist_ok=True)

    img_info = []
    for image_info in image_infos:
        index = 1
        src_img_path = osp.join(root_path, 'imgs', image_info['file_name'])
        image = mmcv.imread(src_img_path)
        src_img_root = image_info['file_name'].split('.')[0]

        for anno in image_info['anno_info']:
            word = anno['word']
            dst_img = crop_img(image, anno['bbox'], 0, 0)

            # Skip invalid annotations
            if min(dst_img.shape) == 0:
                continue

            dst_img_name = f'{src_img_root}_{index}.png'
            index += 1
            dst_img_path = osp.join(dst_image_root, dst_img_name)
            mmcv.imwrite(dst_img, dst_img_path)

            img_info.append({
                'file_name': dst_img_name,
                'anno_info': [{
                    'text': word
                }]
            })

    dump_ocr_data(img_info, dst_label_file, 'textrecog')


def main():
    args = parse_args()
    root_path = args.root_path

    for split in ['train', 'val', 'test']:
        print(f'Processing {split} set...')
        with mmengine.Timer(
                print_tmpl='It takes {}s to convert IMGUR annotation'):
            anno_infos = collect_imgur_info(
                root_path, f'imgur5k_annotations_{split}.json')
            generate_ann(root_path, split, anno_infos)


if __name__ == '__main__':
    main()
