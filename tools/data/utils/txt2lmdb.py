# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmocr.utils import lmdb_converter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--imglist', '-i', required=True, help='input imglist path')
    parser.add_argument(
        '--output', '-o', required=True, help='output lmdb path')
    parser.add_argument(
        '--batch_size',
        '-b',
        type=int,
        default=10000,
        help='processing batch size, default 10000')
    parser.add_argument(
        '--coding',
        '-c',
        default='utf8',
        help='bytes coding scheme, default utf8')
    parser.add_argument(
        '--lmdb_map_size',
        '-l',
        type=int,
        default=109951162776,
        help='maximum size database may grow to , default 109951162776 bytes')
    opt = parser.parse_args()

    lmdb_converter(
        opt.imglist,
        opt.output,
        batch_size=opt.batch_size,
        coding=opt.coding,
        lmdb_map_size=opt.lmdb_map_size)


if __name__ == '__main__':
    main()
