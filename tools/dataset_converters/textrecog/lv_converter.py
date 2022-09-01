# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

from mmocr.utils import dump_ocr_data


def convert_annotations(root_path, split):
    """Convert original annotations to mmocr format.

    The annotation format is as the following:
        Crops/val/11/1/1.png weighted
        Crops/val/11/1/2.png 26
        Crops/val/11/1/3.png casting
        Crops/val/11/1/4.png 28
    After this module, the annotation has been changed to the format below:
        jsonl:
        {'filename': 'Crops/val/11/1/1.png', 'text': 'weighted'}
        {'filename': 'Crops/val/11/1/1.png', 'text': '26'}
        {'filename': 'Crops/val/11/1/1.png', 'text': 'casting'}
        {'filename': 'Crops/val/11/1/1.png', 'text': '28'}

    Args:
        root_path (str): The root path of the dataset
        split (str): The split of dataset. Namely: training or test
    """
    assert isinstance(root_path, str)
    assert isinstance(split, str)

    img_info = []
    with open(
            osp.join(root_path, f'{split}_label.txt'),
            encoding='"utf-8-sig') as f:
        annos = f.readlines()
    for anno in annos:
        if anno:
            # Text may contain spaces
            dst_img_name, word = anno.split('png ')
            word = word.strip('\n')
            img_info.append({
                'file_name': dst_img_name + 'png',
                'anno_info': [{
                    'text': word
                }]
            })
    dump_ocr_data(img_info, osp.join(root_path, f'{split.lower()}_label.json'),
                  'textrecog')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and test set of Lecture Video DB')
    parser.add_argument('root_path', help='Root dir path of Lecture Video DB')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path

    for split in ['train', 'val', 'test']:
        convert_annotations(root_path, split)
        print(f'{split} split converted.')


if __name__ == '__main__':
    main()
