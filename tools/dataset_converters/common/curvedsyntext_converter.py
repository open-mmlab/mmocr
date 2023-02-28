# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from functools import partial

import mmengine
import numpy as np

from mmocr.utils import bezier2polygon, sort_points

# The default dictionary used by CurvedSynthText
dict95 = [
    ' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.',
    '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=',
    '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[',
    '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
    'z', '{', '|', '}', '~'
]
UNK = len(dict95)
EOS = UNK + 1


def digit2text(rec):
    res = []
    for d in rec:
        assert d <= EOS
        if d == EOS:
            break
        if d == UNK:
            print('Warning: Has a UNK character')
            res.append('口')  # Or any special character not in the target dict
        res.append(dict95[d])
    return ''.join(res)


def modify_annotation(ann, num_sample, start_img_id=0, start_ann_id=0):
    ann['text'] = digit2text(ann.pop('rec'))
    # Get hide egmentation points
    polygon_pts = bezier2polygon(ann['bezier_pts'], num_sample=num_sample)
    ann['segmentation'] = np.asarray(sort_points(polygon_pts)).reshape(
        1, -1).tolist()
    ann['image_id'] += start_img_id
    ann['id'] += start_ann_id
    return ann


def modify_image_info(image_info, path_prefix, start_img_id=0):
    image_info['file_name'] = osp.join(path_prefix, image_info['file_name'])
    image_info['id'] += start_img_id
    return image_info


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert CurvedSynText150k to COCO format')
    parser.add_argument('root_path', help='CurvedSynText150k  root path')
    parser.add_argument('-o', '--out-dir', help='Output path')
    parser.add_argument(
        '-n',
        '--num-sample',
        type=int,
        default=4,
        help='Number of sample points at each Bezier curve.')
    parser.add_argument(
        '--nproc', default=1, type=int, help='Number of processes')
    args = parser.parse_args()
    return args


def convert_annotations(data,
                        path_prefix,
                        num_sample,
                        nproc,
                        start_img_id=0,
                        start_ann_id=0):
    modify_image_info_with_params = partial(
        modify_image_info, path_prefix=path_prefix, start_img_id=start_img_id)
    modify_annotation_with_params = partial(
        modify_annotation,
        num_sample=num_sample,
        start_img_id=start_img_id,
        start_ann_id=start_ann_id)
    if nproc > 1:
        data['annotations'] = mmengine.track_parallel_progress(
            modify_annotation_with_params, data['annotations'], nproc=nproc)
        data['images'] = mmengine.track_parallel_progress(
            modify_image_info_with_params, data['images'], nproc=nproc)
    else:
        data['annotations'] = mmengine.track_progress(
            modify_annotation_with_params, data['annotations'])
        data['images'] = mmengine.track_progress(
            modify_image_info_with_params,
            data['images'],
        )
    data['categories'] = [{'id': 1, 'name': 'text'}]
    return data


def main():
    args = parse_args()
    root_path = args.root_path
    out_dir = args.out_dir if args.out_dir else root_path
    mmengine.mkdir_or_exist(out_dir)

    anns = mmengine.load(osp.join(root_path, 'train1.json'))
    data1 = convert_annotations(anns, 'syntext_word_eng', args.num_sample,
                                args.nproc)

    # Get the maximum image id from data1
    start_img_id = max(data1['images'], key=lambda x: x['id'])['id'] + 1
    start_ann_id = max(data1['annotations'], key=lambda x: x['id'])['id'] + 1
    anns = mmengine.load(osp.join(root_path, 'train2.json'))
    data2 = convert_annotations(
        anns,
        'emcs_imgs',
        args.num_sample,
        args.nproc,
        start_img_id=start_img_id,
        start_ann_id=start_ann_id)

    data1['images'] += data2['images']
    data1['annotations'] += data2['annotations']
    mmengine.dump(data1, osp.join(out_dir, 'instances_training.json'))


if __name__ == '__main__':
    main()
