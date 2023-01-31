# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os.path as osp
import time

import lmdb
import mmcv
import numpy as np
from scipy.io import loadmat
from shapely.geometry import Polygon

import mmengine
from mmocr.utils import check_argument


def trace_boundary(char_boxes):
    """Trace the boundary point of text.

    Args:
        char_boxes (list[ndarray]): The char boxes for one text. Each element
        is 4x2 ndarray.

    Returns:
        boundary (ndarray): The boundary point sets with size nx2.
    """
    assert check_argument.is_type_list(char_boxes, np.ndarray)

    # from top left to to right
    p_top = [box[0:2] for box in char_boxes]
    # from bottom right to bottom left
    p_bottom = [
        char_boxes[idx][[2, 3], :]
        for idx in range(len(char_boxes) - 1, -1, -1)
    ]

    p = p_top + p_bottom

    boundary = np.concatenate(p).astype(int)

    return boundary


def match_bbox_char_str(bboxes, char_bboxes, strs):
    """match the bboxes, char bboxes, and strs.

    Args:
        bboxes (ndarray): The text boxes of size (2, 4, num_box).
        char_bboxes (ndarray): The char boxes of size (2, 4, num_char_box).
        strs (ndarray): The string of size (num_strs,)
    """
    assert isinstance(bboxes, np.ndarray)
    assert isinstance(char_bboxes, np.ndarray)
    assert isinstance(strs, np.ndarray)
    bboxes = bboxes.astype(np.int32)
    char_bboxes = char_bboxes.astype(np.int32)

    if len(char_bboxes.shape) == 2:
        char_bboxes = np.expand_dims(char_bboxes, axis=2)
    char_bboxes = np.transpose(char_bboxes, (2, 1, 0))
    if len(bboxes.shape) == 2:
        bboxes = np.expand_dims(bboxes, axis=2)
    bboxes = np.transpose(bboxes, (2, 1, 0))
    chars = ''.join(strs).replace('\n', '').replace(' ', '')
    num_boxes = bboxes.shape[0]

    poly_list = [Polygon(bboxes[iter]) for iter in range(num_boxes)]
    poly_box_list = [bboxes[iter] for iter in range(num_boxes)]

    poly_char_list = [[] for iter in range(num_boxes)]
    poly_char_idx_list = [[] for iter in range(num_boxes)]
    poly_charbox_list = [[] for iter in range(num_boxes)]

    words = []
    for s in strs:
        words += s.split()
    words_len = [len(w) for w in words]
    words_end_inx = np.cumsum(words_len)
    start_inx = 0
    for word_inx, end_inx in enumerate(words_end_inx):
        for char_inx in range(start_inx, end_inx):
            poly_char_idx_list[word_inx].append(char_inx)
            poly_char_list[word_inx].append(chars[char_inx])
            poly_charbox_list[word_inx].append(char_bboxes[char_inx])
        start_inx = end_inx

    for box_inx in range(num_boxes):
        assert len(poly_charbox_list[box_inx]) > 0

    poly_boundary_list = []
    for item in poly_charbox_list:
        boundary = np.ndarray((0, 2))
        if len(item) > 0:
            boundary = trace_boundary(item)
        poly_boundary_list.append(boundary)

    return (poly_list, poly_box_list, poly_boundary_list, poly_charbox_list,
            poly_char_idx_list, poly_char_list)


def convert_annotations(root_path, gt_name, lmdb_name):
    """Convert the annotation into lmdb dataset.

    Args:
        root_path (str): The root path of dataset.
        gt_name (str): The ground truth filename.
        lmdb_name (str): The output lmdb filename.
    """
    assert isinstance(root_path, str)
    assert isinstance(gt_name, str)
    assert isinstance(lmdb_name, str)
    start_time = time.time()
    gt = loadmat(gt_name)
    img_num = len(gt['imnames'][0])
    env = lmdb.open(lmdb_name, map_size=int(1e9 * 40))
    with env.begin(write=True) as txn:
        for img_id in range(img_num):
            if img_id % 1000 == 0 and img_id > 0:
                total_time_sec = time.time() - start_time
                avg_time_sec = total_time_sec / img_id
                eta_mins = (avg_time_sec * (img_num - img_id)) / 60
                print(f'\ncurrent_img/total_imgs {img_id}/{img_num} | '
                      f'eta: {eta_mins:.3f} mins')
            # for each img
            img_file = osp.join(root_path, 'imgs', gt['imnames'][0][img_id][0])
            img = mmcv.imread(img_file, 'unchanged')
            height, width = img.shape[0:2]
            img_json = {}
            img_json['file_name'] = gt['imnames'][0][img_id][0]
            img_json['height'] = height
            img_json['width'] = width
            img_json['annotations'] = []
            wordBB = gt['wordBB'][0][img_id]
            charBB = gt['charBB'][0][img_id]
            txt = gt['txt'][0][img_id]
            poly_list, _, poly_boundary_list, _, _, _ = match_bbox_char_str(
                wordBB, charBB, txt)
            for poly_inx in range(len(poly_list)):

                polygon = poly_list[poly_inx]
                min_x, min_y, max_x, max_y = polygon.bounds
                bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
                anno_info = dict()
                anno_info['iscrowd'] = 0
                anno_info['category_id'] = 1
                anno_info['bbox'] = bbox
                anno_info['segmentation'] = [
                    poly_boundary_list[poly_inx].flatten().tolist()
                ]

                img_json['annotations'].append(anno_info)
            string = json.dumps(img_json)
            txn.put(str(img_id).encode('utf8'), string.encode('utf8'))
        key = b'total_number'
        value = str(img_num).encode('utf8')
        txn.put(key, value)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert synthtext to lmdb dataset')
    parser.add_argument('synthtext_path', help='synthetic root path')
    parser.add_argument('-o', '--out-dir', help='output path')
    args = parser.parse_args()
    return args


# TODO: Refactor synthtext
def main():
    args = parse_args()
    synthtext_path = args.synthtext_path
    out_dir = args.out_dir if args.out_dir else synthtext_path
    mmengine.mkdir_or_exist(out_dir)

    gt_name = osp.join(synthtext_path, 'gt.mat')
    lmdb_name = 'synthtext.lmdb'
    convert_annotations(synthtext_path, gt_name, osp.join(out_dir, lmdb_name))


if __name__ == '__main__':
    main()
