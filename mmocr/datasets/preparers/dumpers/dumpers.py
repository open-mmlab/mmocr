# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Dict, List

import cv2
import lmdb
import mmengine
import numpy as np

from mmocr.utils import list_to_file
from ..data_preparer import DATA_DUMPERS


@DATA_DUMPERS.register_module()
class JsonDumper:

    def __init__(self, task: str) -> None:
        self.task = task

    def dump(self, data: Dict, data_root: str, split: str) -> None:
        """Dump data to json file.

        Args:
            data (Dict): Data to be dumped.
            data_root (str): Root directory of data.
            split (str): Split of data.
            cfg_path (str): Path to configs. Defaults to 'configs/'.
        """

        filename = f'{self.task}_{split}.json'
        dst_file = osp.join(data_root, filename)
        mmengine.dump(data, dst_file, ensure_ascii=False)


@DATA_DUMPERS.register_module()
class WildreceiptOpensetDumper:

    def __init__(self, task: str) -> None:
        self.task = task

    def dump(self, data: List, data_root: str, split: str):
        """Dump data to txt file.

        Args:
            data (List): Data to be dumped.
            data_root (str): Root directory of data.
            split (str): Split of data.
        """

        filename = f'openset_{split}.txt'
        dst_file = osp.join(data_root, filename)
        list_to_file(dst_file, data)


@DATA_DUMPERS.register_module()
class LMDBDumper:
    """Text recognition LMDB format dataset dumper.

    Args:
        task (str): Task type.
        batch_size (int): Number of files written to the cache each time.
        encoding (str): Label encoding method.
        lmdb_map_size (int): Maximum size database may grow to.
        verify (bool): If true, check the validity of
            every image.Defaults to True.
    """

    def __init__(self,
                 task: str,
                 batch_size: int = 1000,
                 encoding: str = 'utf-8',
                 lmdb_map_size: int = 1099511627776,
                 verify: bool = True) -> None:
        assert task == 'textrecog', \
            f'LMDBDumper only works with textrecog, but got {task}'
        self.task = task
        self.batch_size = batch_size
        self.encoding = encoding
        self.lmdb_map_size = lmdb_map_size
        self.verify = verify

    def check_image_is_valid(self, imageBin):
        if imageBin is None:
            return False
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
        if imgH * imgW == 0:
            return False
        return True

    def write_cache(self, env, cache):
        with env.begin(write=True) as txn:
            cursor = txn.cursor()
            cursor.putmulti(cache, dupdata=False, overwrite=True)

    def parser_pack_instance(self, instance: Dict):
        """parser an packed MMOCR format textrecog instance.

        Args:
            instance (Dict): An packed MMOCR format textrecog instance.
                For example,
                {
                    "instance": [
                        {
                            "text": "Hello"
                        }
                    ],
                    "img_path": "img1.jpg"
                }
        """
        assert isinstance(instance,
                          Dict), 'Element of data_list must be a dict'
        assert 'img_path' in instance and 'instances' in instance, \
            'Element of data_list must have the following keys: ' \
            f'img_path and instances, but got {instance.keys()}'
        assert isinstance(instance['instances'], List) and len(
            instance['instances']) == 1
        assert 'text' in instance['instances'][0]

        img_path = instance['img_path']
        text = instance['instances'][0]['text']
        return img_path, text

    def dump(self, data: Dict, data_root: str, split: str) -> None:
        """Dump data to LMDB format."""

        # create lmdb env
        output_dirname = f'{self.task}_{split}.lmdb'
        output = osp.join(data_root, output_dirname)
        mmengine.mkdir_or_exist(output)
        env = lmdb.open(output, map_size=self.lmdb_map_size)
        # load data
        if 'data_list' not in data:
            raise ValueError('Dump data must have data_list key')
        data_list = data['data_list']
        cache = []
        # index start from 1
        cnt = 1
        n_samples = len(data_list)
        for d in data_list:
            # convert both images and labels to lmdb
            label_key = 'label-%09d'.encode(self.encoding) % cnt
            img_name, text = self.parser_pack_instance(d)
            img_path = osp.join(data_root, img_name)
            if not osp.exists(img_path):
                warnings.warn('%s does not exist' % img_path)
                continue
            with open(img_path, 'rb') as f:
                image_bin = f.read()
            if self.verify:
                if not self.check_image_is_valid(image_bin):
                    warnings.warn('%s is not a valid image' % img_path)
                    continue
            image_key = 'image-%09d'.encode(self.encoding) % cnt
            cache.append((image_key, image_bin))
            cache.append((label_key, text.encode(self.encoding)))

            if cnt % self.batch_size == 0:
                self.write_cache(env, cache)
                cache = []
                print('Written %d / %d' % (cnt, n_samples))
            cnt += 1
        n_samples = cnt - 1
        cache.append(('num-samples'.encode(self.encoding),
                      str(n_samples).encode(self.encoding)))
        self.write_cache(env, cache)
        print('Created lmdb dataset with %d samples' % n_samples)
