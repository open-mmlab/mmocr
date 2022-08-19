# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import json
import os.path as osp
import warnings
from functools import partial

import mmcv
import mmengine

from mmocr.utils import list_to_file
from mmocr.utils.img_utils import crop_img, warp_img


def parse_labelme_json(json_file,
                       img_dir,
                       out_dir,
                       tasks,
                       ignore_marker='###',
                       recog_format='jsonl',
                       warp_flag=False):
    invalid_res = [[], [], []]

    json_obj = mmengine.load(json_file)

    img_file = osp.basename(json_obj['imagePath'])
    img_full_path = osp.join(img_dir, img_file)

    img_width = json_obj['imageWidth']
    img_height = json_obj['imageHeight']
    if 'recog' in tasks:
        src_img = mmcv.imread(img_full_path)
        img_basename = osp.splitext(img_file)[0]
        sub_dir = osp.join(out_dir, 'crops', img_basename)
        mmcv.mkdir_or_exist(sub_dir)

    det_line_json_list = []
    recog_crop_line_str_list = []
    recog_warp_line_str_list = []

    shape_info = json_obj['shapes']
    idx = 0
    annos = []
    for box_info in shape_info:
        shape = box_info['shape_type']
        if shape not in ['rectangle', 'polygon']:
            msg = 'Only \'rectangle\' and \'polygon\' boxes are supported. '
            msg += f'Boxes with {shape} will be discarded.'
            warnings.warn(msg)
            return invalid_res
        poly = []
        box_points = box_info['points']
        for point in box_points:
            poly.extend([int(x) for x in point])
        x_list = poly[0::2]
        y_list = poly[1::2]
        quad = []
        if shape == 'rectangle':
            warp_flag = False
            quad = [
                poly[0], poly[1], poly[2], poly[1], poly[2], poly[3], poly[0],
                poly[3]
            ]
        else:
            if len(poly) < 8 or len(poly) % 2 != 0:
                msg = f'Invalid polygon {poly}. '
                msg += 'The polygon is expected to have 8 or more than 8 '
                msg += 'even number of coordinates in MMOCR.'
                warnings.warn(msg)
                return invalid_res
            if len(poly) == 8:
                quad = poly
            else:
                warp_flag = False
                x_min, x_max, y_min, y_max = min(x_list), max(x_list), min(
                    y_list), max(y_list)
                quad = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
        text_label = box_info['label']
        # for textdet
        anno = {}
        anno['iscrowd'] = 0 if text_label != ignore_marker else 1
        anno['category_id'] = 1
        w = max(x_list) - min(x_list)
        h = max(y_list) - min(y_list)
        anno['bbox'] = [min(x_list), min(y_list), w, h]
        if shape == 'rectangle':
            anno['segmentation'] = [quad]
        else:
            anno['segmentation'] = [poly]
        anno['text'] = text_label
        annos.append(anno)
        # for textrecog
        if 'recog' in tasks:
            if text_label == ignore_marker or len(text_label) == 0:
                continue
            cropped_img = crop_img(src_img, quad)
            img_path_cropped_img = osp.join(sub_dir, f'crop_{idx}.jpg')
            mmcv.imwrite(cropped_img, img_path_cropped_img)
            if recog_format == 'txt':
                recog_crop_line_str_list.append(
                    f'{img_path_cropped_img} {text_label}')
            elif recog_format == 'jsonl':
                recog_crop_line_str_list.append(
                    json.dumps({
                        'filename': img_path_cropped_img,
                        'text': text_label
                    }))
            else:
                raise NotImplementedError
            if warp_flag:
                warpped_img = warp_img(src_img, quad)
                img_path_warpped_img = osp.join(sub_dir, f'warp_{idx}.jpg')
                mmcv.imwrite(warpped_img, img_path_warpped_img)
                if recog_format == 'txt':
                    recog_warp_line_str_list.append(
                        f'{img_path_warpped_img} {text_label}')
                elif recog_format == 'jsonl':
                    recog_warp_line_str_list.append(
                        json.dumps({
                            'filename': img_path_warpped_img,
                            'text': text_label
                        }))
        idx += 1

    line_json = {
        'file_name': img_file,
        'height': img_height,
        'width': img_width,
        'annotations': annos
    }
    det_line_json_list.append(json.dumps(line_json, ensure_ascii=False))

    return [
        det_line_json_list, recog_crop_line_str_list, recog_warp_line_str_list
    ]


def process(json_dir,
            img_dir,
            out_dir,
            tasks=['det'],
            nproc=1,
            recog_format='jsonl',
            warp=False):
    mmcv.mkdir_or_exist(out_dir)

    json_file_list = glob.glob(osp.join(json_dir, '*.json'))

    parse_labelme_json_func = partial(
        parse_labelme_json,
        img_dir=img_dir,
        out_dir=out_dir,
        tasks=tasks,
        recog_format=recog_format,
        warp_flag=warp)

    if nproc <= 1:
        total_results = mmcv.track_progress(parse_labelme_json_func,
                                            json_file_list)
    else:
        total_results = mmcv.track_parallel_progress(
            parse_labelme_json_func,
            json_file_list,
            keep_order=True,
            nproc=nproc)

    total_det_line_json_list = []
    total_recog_crop_line_str = []
    total_recog_warp_line_str = []
    for res in total_results:
        total_det_line_json_list.extend(res[0])
        if 'recog' in tasks:
            total_recog_crop_line_str.extend(res[1])
            total_recog_warp_line_str.extend(res[2])

    mmcv.mkdir_or_exist(out_dir)
    det_out_file = osp.join(out_dir, 'instances_training.txt')
    list_to_file(det_out_file, total_det_line_json_list)

    if 'recog' in tasks:
        recog_out_file_crop = osp.join(out_dir, f'train_label.{recog_format}')
        list_to_file(recog_out_file_crop, total_recog_crop_line_str)
        if warp:
            recog_out_file_warp = osp.join(out_dir,
                                           f'warp_train_label.{recog_format}')
            list_to_file(recog_out_file_warp, total_recog_warp_line_str)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_dir', help='Root dir for labelme json file.')
    parser.add_argument('image_dir', help='Root dir for image file.')
    parser.add_argument(
        'out_dir', help='Dir to save annotations in mmocr format.')
    parser.add_argument(
        '--tasks',
        nargs='+',
        help='Tasks to be processed, can be only "det" or both: "det recog"')
    parser.add_argument(
        '--nproc', type=int, default=1, help='Number of process.')
    parser.add_argument(
        '--format',
        default='jsonl',
        help='Use jsonl or string to format recognition annotations',
        choices=['jsonl', 'txt'])
    parser.add_argument(
        '--warp',
        help='Store warpped img for recognition task',
        action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    process(args.json_dir, args.image_dir, args.out_dir, args.tasks,
            args.nproc, args.format, args.warp)

    print('finish')


if __name__ == '__main__':
    main()
