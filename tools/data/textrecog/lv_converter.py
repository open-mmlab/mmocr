# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os.path as osp

from mmocr.utils.fileio import list_to_file


def convert_annotations(root_path, split, format):
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
        format (str): Annotation format, should be either 'txt' or 'jsonl'
    """
    assert isinstance(root_path, str)
    assert isinstance(split, str)

    if format == 'txt':  # LV has already provided txt format annos
        return

    if format == 'jsonl':
        lines = []
        with open(
                osp.join(root_path, f'{split}_label.txt'),
                encoding='"utf-8-sig') as f:
            annos = f.readlines()
        for anno in annos:
            if anno:
                # Text may contain spaces
                dst_img_name, word = anno.split('png ')
                word = word.strip('\n')
                lines.append(
                    json.dumps({
                        'filename': dst_img_name + 'png',
                        'text': word
                    }))
    else:
        raise NotImplementedError

    list_to_file(osp.join(root_path, f'{split}_label.{format}'), lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and test set of Lecture Video DB')
    parser.add_argument('root_path', help='Root dir path of Lecture Video DB')
    parser.add_argument(
        '--format',
        default='jsonl',
        help='Use jsonl or string to format annotations',
        choices=['jsonl', 'txt'])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path

    for split in ['train', 'val', 'test']:
        convert_annotations(root_path, split, args.format)
        print(f'{split} split converted.')


if __name__ == '__main__':
    main()
