# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmocr.utils import recog2lmdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('label_path', type=str, help='Path to label file')
    parser.add_argument('output', type=str, help='Output lmdb path')
    parser.add_argument(
        '--img-root', '-i', type=str, help='Input imglist path')
    parser.add_argument(
        '--label-only',
        action='store_true',
        help='Only converter label to lmdb')
    parser.add_argument(
        '--label-format',
        '-f',
        default='txt',
        choices=['txt', 'jsonl'],
        help='The format of the label file, either txt or jsonl')
    parser.add_argument(
        '--batch-size',
        '-b',
        type=int,
        default=1000,
        help='Processing batch size, defaults to 1000')
    parser.add_argument(
        '--encoding',
        '-e',
        type=str,
        default='utf8',
        help='Bytes coding scheme, defaults to utf8')
    parser.add_argument(
        '--lmdb-map-size',
        '-m',
        type=int,
        default=1099511627776,
        help='Maximum size database may grow to, '
        'defaults to 1099511627776 bytes (1TB)')
    opt = parser.parse_args()

    assert opt.img_root or opt.label_only
    recog2lmdb(opt.img_root, opt.label_path, opt.output, opt.label_format,
               opt.label_only, opt.batch_size, opt.encoding, opt.lmdb_map_size)


if __name__ == '__main__':
    main()
