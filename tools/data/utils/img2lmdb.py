# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
import os.path as osp

import cv2
import lmdb
import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def parseline(line, format):
    if format == 'txt':
        img_name, text = line.strip('\n').split(' ')
    else:
        img_name = line['filename']
        text = line['text']
    return img_name, text


def img2lmdb(imgs_path,
             label_path,
             label_format,
             output,
             batch_size,
             coding,
             lmdb_map_size,
             checkValid=True):
    """Create LMDB dataset with images and labels for text recognition.

    Args:
        imgs_path (str): path to images
        label_path (str): path to label file
        label_format (str): format of the label file, either txt or jsonl
        output (str): LMDB output path.
        batch_size (int): Number of files written to the cache each time.
        coding (str): label encoding method.
        lmdb_map_size (int): maximum size database may grow to
        checkValid (bool, optional): if true, check the validity of
            every image.Defaults to True.
    E.g.
    This function supports MMOCR's recognition data format and the label file
    can be txt or jsonl, as follows:

        ├──img_path
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
    # creat lmdb env
    os.makedirs(output, exist_ok=True)
    env = lmdb.open(output, map_size=lmdb_map_size)
    # load label file
    if label_format == 'jsonl':
        anno_list = json.load(open(label_path, 'r', encoding=coding))
    else:
        with open(label_path, 'r', encoding=coding) as f:
            anno_list = f.readlines()

    cache = {}
    # index start from 1
    cnt = 1
    nSamples = len(anno_list)
    for anno in anno_list:
        img_name, text = parseline(anno, label_format)
        img_path = osp.join(imgs_path, img_name)
        if not osp.exists(img_path):
            print('%s does not exist' % img_path)
            continue
        with open(img_path, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % img_path)
                    continue
            except Exception:
                print('error occured at ', img_name)

        imageKey = 'image-%09d'.encode(coding) % cnt
        labelKey = 'label-%09d'.encode(coding) % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = text.encode(coding)

        if cnt % batch_size == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'.encode(coding)] = str(nSamples).encode(coding)
    writeCache(env, cache)
    print('Created lmdb dataset with %d samples' % nSamples)


def main():
    parser = argparse.ArgumentParser(
        description='Convert recognition dataset in MMOCR form to lmdb form')
    parser.add_argument(
        '--imgs-path', '-i', required=True, help='Path to images')
    parser.add_argument(
        '--label-path', '-l', required=True, help='Path to label file')
    parser.add_argument(
        '--label-format',
        '-f',
        default='txt',
        choices=['txt', 'jsonl'],
        help='The format of the label file, either txt or jsonl')
    parser.add_argument(
        '--output', '-o', required=True, help='output lmdb path')
    parser.add_argument(
        '--batch-size',
        '-b',
        type=int,
        default=1000,
        help='processing batch size, default 1000')
    parser.add_argument(
        '--coding',
        '-c',
        default='utf-8',
        help='bytes coding scheme, default utf8')
    parser.add_argument(
        '--lmdb-map-size',
        '-l',
        type=int,
        default=109951162776,
        help='maximum size database may grow to , default 109951162776 bytes')
    opt = parser.parse_args()
    img2lmdb(opt.img_path, opt.label_path, opt.label_format, opt.output,
             opt.batch_size, opt.coding, opt.lmdb_map_size)
