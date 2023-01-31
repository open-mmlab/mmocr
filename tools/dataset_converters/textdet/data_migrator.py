# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List

import mmengine
from mmocr.utils import dump_ocr_data


def parse_coco_json(in_path: str) -> List[Dict]:
    """Load coco annotations into image_infos parsable by dump_ocr_data().

    Args:
        in_path (str): COCO text annotation path.

    Returns:
        list[dict]: List of image information dicts. To be used by
        dump_ocr_data().
    """
    json_obj = mmengine.load(in_path)
    image_infos = json_obj['images']
    annotations = json_obj['annotations']
    imgid2annos = defaultdict(list)
    for anno in annotations:
        new_anno = deepcopy(anno)
        new_anno['category_id'] = 0  # Must be 0 for OCR tasks which stands
        # for "text" category
        imgid2annos[anno['image_id']].append(new_anno)

    results = []
    for image_info in image_infos:
        image_info['anno_info'] = imgid2annos[image_info['id']]
        results.append(image_info)

    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', help='Input json path in coco format.')
    parser.add_argument(
        'out_path', help='Output json path in openmmlab format.')
    parser.add_argument(
        '--task',
        type=str,
        default='auto',
        choices=['auto', 'textdet', 'textspotter'],
        help='Output annotation type, defaults to "auto", which decides the'
        'best task type based on whether "text" is annotated. Other options'
        'are "textdet" and "textspotter".')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    image_infos = parse_coco_json(args.in_path)
    task_name = args.task
    if task_name == 'auto':
        task_name = 'textdet'
        if 'text' in image_infos[0]['anno_info'][0]:
            task_name = 'textspotter'
    dump_ocr_data(image_infos, args.out_path, task_name)
    print('finish')


if __name__ == '__main__':
    main()
