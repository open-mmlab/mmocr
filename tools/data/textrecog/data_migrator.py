# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
from typing import List, Tuple

from mmocr.datasets.utils.loader import AnnFileLoader
from mmocr.utils import StringStrip, dump_ocr_data, recog_anno_to_imginfo


def parse_legacy_data(in_path: str,
                      format: str) -> Tuple[List[str], List[str]]:
    """Load legacy data and return a list of file paths and labels.

    Args:
        in_path (str): Path to annotation file.
        format (str): Annotation format. Choices are 'txt', 'json' and 'lmdb'.

    Returns:
        tuple(list[str], list[str]): File paths and labels.
    """
    file_paths = []
    labels = []
    strip_cls = StringStrip()
    if format == 'lmdb':
        # TODO: Backend might be deprecated
        loader = AnnFileLoader(
            in_path,
            parser=dict(type='LineJsonParser', keys=['filename', 'text']),
            file_format='lmdb')
        num = len(loader)
        for i in range(num):
            line_json = loader[i]
            file_path, label = line_json['filename'], line_json['text']
            file_path = strip_cls(file_path)
            label = strip_cls(label)
            file_paths.append(file_path)
            labels.append(label)
        return file_paths, labels
    else:
        with open(in_path) as f:
            if format == 'txt':
                for line in f:
                    line = strip_cls(line)
                    file_path, label = line.split()[:2]
                    file_paths.append(file_path)
                    labels.append(label)
            elif format == 'jsonl':
                for line in f:
                    datum = json.loads(line)
                    file_paths.append(datum['filename'])
                    labels.append(datum['text'])

    return file_paths, labels


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Convert annotations for'
        'text recognition tasks in MMOCR 0.x into the latest openmmlab format.'
    )
    parser.add_argument(
        'in_path', help='The path to legacy recognition data file')
    parser.add_argument(
        'out_path', help='The output json path in openmmlab format')
    parser.add_argument(
        '--format',
        choices=['txt', 'jsonl', 'lmdb'],
        type=str,
        default='txt',
        help='Legacy data format')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    file_paths, labels = parse_legacy_data(args.in_path, args.format)
    img_infos = recog_anno_to_imginfo(file_paths, labels)
    dump_ocr_data(img_infos, args.out_path, 'textrecog')
    print('finish')


if __name__ == '__main__':
    main()
