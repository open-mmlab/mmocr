# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os.path as osp

from mmocr.utils.fileio import list_to_file


def convert_annotations(root_path, split, format):
    """Convert original annotations to mmocr format
    The annotation format is as the following:
        word_1.png, "flying"
        word_2.png, "today"
        word_3.png, "means"
    After this module, the annotation has been changed to the format below:
        txt:
        word_1.png flying
        word_2.png today
        word_3.png means

        jsonl:
        {'filename': 'word_1.png', 'text': 'flying'}
        {'filename': 'word_2.png', 'text': 'today'}
        {'filename': 'word_3.png', 'text': 'means'}

    Args:
        root_path (str): The root path of the dataset
        split (str): The split of dataset. Namely: training or test

    """
    assert isinstance(root_path, str)
    assert isinstance(split, str)

    lines = []
    with open(
            osp.join(root_path, 'annotations',
                     f'Challenge1_{split}_Task3_GT.txt'),
            'r',
            encoding='"utf-8-sig') as f:
        annos = f.readlines()
    dst_image_root = osp.join(root_path, split)
    for anno in annos:
        # text may contain comma ','
        dst_img_name, word = anno.split(', "')
        word = word.replace('"\n', '')
        if format == 'txt':
            lines.append(f'{osp.basename(dst_image_root)}/{dst_img_name} '
                         f'{word}')
        elif format == 'jsonl':
            lines.append(
                json.dumps({
                    'filename':
                    f'{osp.basename(dst_image_root)}/{dst_img_name}',
                    'text': word
                }))
        else:
            raise NotImplementedError

    list_to_file(osp.join(root_path, f'{split}_label.txt'), lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and test set of IC11')
    parser.add_argument('root_path', help='Root dir path of IC11')
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

    for split in ['Train', 'Test']:
        convert_annotations(root_path, split, args.format)
        print(f'{split} split converted.')


if __name__ == '__main__':
    main()
