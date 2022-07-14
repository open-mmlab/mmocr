# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
import os.path as osp

import cv2
import lmdb
import numpy as np

from mmocr.utils import list_from_file


def parse_line(line, format):
    if format == 'txt':
        img_name, text = line.split(' ')
    else:
        line = json.loads(line)
        img_name = line['filename']
        text = line['text']
    return img_name, text


def check_image_is_valid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def write_cache(env, cache):
    with env.begin(write=True) as txn:
        cursor = txn.cursor()
        cursor.putmulti(cache, dupdata=False, overwrite=True)


def recog2lmdb(img_root,
               label_path,
               output,
               label_format='txt',
               label_only=False,
               batch_size=1000,
               encoding='utf-8',
               lmdb_map_size=109951162776,
               verify=True):
    """Create text recognition dataset to LMDB format.

    Args:
        img_root (str): Path to images.
        label_path (str): Path to label file.
        output (str): LMDB output path.
        label_format (str): Format of the label file, either txt or jsonl.
        label_only (bool): Only convert label to lmdb format.
        batch_size (int): Number of files written to the cache each time.
        encoding (str): Label encoding method.
        lmdb_map_size (int): Maximum size database may grow to.
        verify (bool): If true, check the validity of
            every image.Defaults to True.

    E.g.
    This function supports MMOCR's recognition data format and the label file
    can be txt or jsonl, as follows:

        ├──img_root
        |      |—— img1.jpg
        |      |—— img2.jpg
        |      |—— ...
        |——label.txt (or label.jsonl)

        label.txt: img1.jpg HELLO
                   img2.jpg WORLD
                   ...

        label.jsonl: {'filename':'img1.jpg', 'text':'HELLO'}
                     {'filename':'img2.jpg', 'text':'WORLD'}
                     ...
    """
    # check label format
    assert osp.basename(label_path).split('.')[-1] == label_format
    # create lmdb env
    os.makedirs(output, exist_ok=True)
    env = lmdb.open(output, map_size=lmdb_map_size)
    # load label file
    anno_list = list_from_file(label_path, encoding=encoding)
    cache = []
    # index start from 1
    cnt = 1
    n_samples = len(anno_list)
    for anno in anno_list:
        label_key = 'label-%09d'.encode(encoding) % cnt
        img_name, text = parse_line(anno, label_format)
        if label_only:
            # convert only labels to lmdb
            line = json.dumps(
                dict(filename=img_name, text=text), ensure_ascii=False)
            cache.append((label_key, line.encode(encoding)))
        else:
            # convert both images and labels to lmdb
            img_path = osp.join(img_root, img_name)
            if not osp.exists(img_path):
                print('%s does not exist' % img_path)
                continue
            with open(img_path, 'rb') as f:
                image_bin = f.read()
            if verify:
                try:
                    if not check_image_is_valid(image_bin):
                        print('%s is not a valid image' % img_path)
                        continue
                except Exception:
                    print('error occurred at ', img_name)
            image_key = 'image-%09d'.encode(encoding) % cnt
            cache.append((image_key, image_bin))
            cache.append((label_key, text.encode(encoding)))

        if cnt % batch_size == 0:
            write_cache(env, cache)
            cache = []
            print('Written %d / %d' % (cnt, n_samples))
        cnt += 1
    n_samples = cnt - 1
    cache.append(
        ('num-samples'.encode(encoding), str(n_samples).encode(encoding)))
    write_cache(env, cache)
    print('Created lmdb dataset with %d samples' % n_samples)


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
        default=109951162776,
        help='Maximum size database may grow to, '
        'defaults to 109951162776 bytes')
    opt = parser.parse_args()

    assert opt.img_root or opt.label_only
    recog2lmdb(opt.img_root, opt.label_path, opt.output, opt.label_format,
               opt.label_only, opt.batch_size, opt.encoding, opt.lmdb_map_size)


if __name__ == '__main__':
    main()
