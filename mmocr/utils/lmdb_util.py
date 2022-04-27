# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import os.path as osp
import shutil
import sys
import time
from pathlib import Path

import cv2
import lmdb
import numpy as np

from mmocr.utils import list_from_file


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
        for k, v in cache.items():
            txn.put(k, v)


def parse_line(line, format):
    if format == 'txt':
        img_name, text = line.strip('\n').split(' ')
    else:
        line = json.loads(line)
        img_name = line['filename']
        text = line['text']
    return [img_name, text]


def label2lmdb(label_path,
               label_format,
               output,
               batch_size=1000,
               encoding='utf-8',
               lmdb_map_size=109951162776):
    """Create LMDB dataset with only labels for text recognition.

    Args:
        label_path (str): path to label file.
        label_format (str): format of the label file, either txt or jsonl.
        output (str): LMDB output path.
        batch_size (int): Number of files written to the cache each time.
        encoding (str): label encoding method.
        lmdb_map_size (int): maximum size database may grow to.
    """
    # read labels
    lines = list_from_file(label_path)

    # create lmdb database
    if Path(output).is_dir():
        while True:
            print('%s already exist, delete or not? [Y/n]' % output)
            Yn = input().strip()
            if Yn in ['Y', 'y']:
                shutil.rmtree(output)
                break
            if Yn in ['N', 'n']:
                return
    print('create database %s' % output)
    Path(output).mkdir(parents=True, exist_ok=False)
    env = lmdb.open(output, map_size=lmdb_map_size)

    # build lmdb
    beg_time = time.strftime('%H:%M:%S')
    for beg_index in range(0, len(lines), batch_size):
        end_index = min(beg_index + batch_size, len(lines))
        sys.stdout.write('\r[%s-%s], processing [%d-%d] / %d' %
                         (beg_time, time.strftime('%H:%M:%S'), beg_index,
                          end_index, len(lines)))
        sys.stdout.flush()
        batch = [(str(index).encode(encoding),
                  ' '.join(parse_line(lines[index],
                                      label_format)).encode(encoding))
                 for index in range(beg_index, end_index)]
        with env.begin(write=True) as txn:
            cursor = txn.cursor()
            cursor.putmulti(batch, dupdata=False, overwrite=True)
    sys.stdout.write('\n')
    with env.begin(write=True) as txn:
        key = 'total_number'.encode(encoding)
        value = str(len(lines)).encode(encoding)
        txn.put(key, value)
    print('done', flush=True)


def img2lmdb(img_root,
             label_path,
             label_format,
             output,
             batch_size=1000,
             encoding='utf-8',
             lmdb_map_size=109951162776,
             verify=True):
    """Create LMDB dataset with images and labels for text recognition. This
    was partially adapted from https://github.com/clovaai/deep-text-
    recognition-benchmark.

    Args:
        img_root (str): path to images.
        label_path (str): path to label file.
        label_format (str): format of the label file, either txt or jsonl.
        output (str): LMDB output path.
        batch_size (int): Number of files written to the cache each time.
        encoding (str): label encoding method.
        lmdb_map_size (int): maximum size database may grow to.
        verify (bool): if true, check the validity of
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
    with open(label_path, 'r', encoding=encoding) as f:
        anno_list = f.readlines()

    cache = {}
    # index start from 1
    cnt = 1
    nSamples = len(anno_list)
    for anno in anno_list:
        img_name, text = parse_line(anno, label_format)
        img_path = osp.join(img_root, img_name)
        if not osp.exists(img_path):
            print('%s does not exist' % img_path)
            continue
        with open(img_path, 'rb') as f:
            imageBin = f.read()
        if verify:
            try:
                if not check_image_is_valid(imageBin):
                    print('%s is not a valid image' % img_path)
                    continue
            except Exception:
                print('error occurred at ', img_name)

        imageKey = 'image-%09d'.encode(encoding) % cnt
        labelKey = 'label-%09d'.encode(encoding) % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = text.encode(encoding)

        if cnt % batch_size == 0:
            write_cache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['total_number'.encode(encoding)] = str(nSamples).encode(encoding)
    write_cache(env, cache)
    print('Created lmdb dataset with %d samples' % nSamples)
