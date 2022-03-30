# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math
import os.path as osp

import mmcv

from mmocr.utils import convert_annotations


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and validation set of LSVT ')
    parser.add_argument('root_path', help='Root dir path of LSVT')
    parser.add_argument(
        '--val-ratio', help='Split ratio for val set', default=0.0, type=float)
    args = parser.parse_args()
    return args


def collect_lsvt_info(root_path, split):
    """Collect the annotation information.

    The annotation format is as the following:
    [
        {'gt_1234': # 'gt_1234' is file name
            [
                {
                    'transcription': '一站式购物中心',
                    'points': [[45, 272], [215, 273], [212, 296], [45, 290]]
                    'illegibility': False
                }, ...
            ]
        }
    ]


    Args:
        root_path (str): Root path to the dataset
        split (str): Dataset split, which should be 'train' or 'val'

    Returns:
        img_info (dict): The dict of the img and annotation information
    """

    annotation_path = osp.join(root_path, 'annotations/train_full_labels.json')
    if not osp.exists(annotation_path):
        raise Exception(
            f'{annotation_path} not exists, please check and try again.')

    annotation = mmcv.load(annotation_path)

    img_infos = []
    for i, img_info in enumerate(annotation['imgs'].values()):
        if img_info['set'] == split:
            img_info['segm_file'] = annotation_path
            ann_ids = annotation['imgToAnns'][str(img_info['id'])]
            # Filter out images without text
            if len(ann_ids) == 0:
                continue
            anno_info = []
            for ann_id in ann_ids:
                ann = annotation['anns'][str(ann_id)]

                # Ignore illegible or non-English words
                iscrowd = 1 if ann['language'] == 'not english' or ann[
                    'legibility'] == 'illegible' else 0

                x, y, w, h = ann['bbox']
                x, y = max(0, math.floor(x)), max(0, math.floor(y))
                w, h = math.ceil(w), math.ceil(h)
                bbox = [x, y, w, h]
                segmentation = [max(0, int(x)) for x in ann['mask']]
                anno = dict(
                    iscrowd=iscrowd,
                    category_id=1,
                    bbox=bbox,
                    area=ann['area'],
                    segmentation=[segmentation])
                anno_info.append(anno)
            img_info.update(anno_info=anno_info)
            img_infos.append(img_info)
    return img_infos


def main():
    args = parse_args()
    root_path = args.root_path
    print('Processing training set...')
    training_infos = collect_lsvt_info(root_path, 'train')
    convert_annotations(training_infos,
                        osp.join(root_path, 'instances_training.json'))
    print('Processing validation set...')
    val_infos = collect_lsvt_info(root_path, 'val')
    convert_annotations(val_infos, osp.join(root_path, 'instances_val.json'))
    print('Finish')


if __name__ == '__main__':
    main()
