# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, List

import json
import cv2
import lmdb
import numpy as np

import mmengine

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

    def __init__(self,
                 task: str,
                 label_only: bool = False,
                 batch_size: int = 1000,
                 encoding: str = 'utf-8',
                 lmdb_map_size: int = 1099511627776,
                 verify: bool = True) -> None:
        assert task == 'textrecog', \
            f'Using LMDBDumper task must be textrecog, but got {task}'
        self.task = task
        self.label_only = label_only
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
        """parser an packed MMOCR format textrecog instance

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
        assert isinstance(instance, Dict), 'Element of data_list must be a dict'
        assert 'img_path' in instance and 'instances' in instance, \
            'Element of data_list must have the following keys: img_path and instances,' \
            f'but got {instance.keys()}'
        assert isinstance(instance['instances'], List) and len(instance['instances']) == 1
        assert 'text' in instance['instances'][0]

        img_path = instance['img_path']
        text = instance['instances'][0]['text']
        return img_path, text

    def dump(self, data: Dict, data_root: str, split: str) -> None:
        """Dump data to LMDB format.

        Args:
            data (Dict): Data to be dumped.
            data_root (str): Root directory of data.
            split (str): Split of data.
            cfg_path (str): Path to configs. Defaults to 'configs/'.
        """

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
            label_key = 'label-%09d'.encode(self.encoding) % cnt
            img_name, text = self.parser_pack_instance(d)
            if self.label_only:
                # convert only labels to lmdb
                line = json.dumps(
                    dict(filename=img_name, text=text), ensure_ascii=False)
                cache.append((label_key, line.encode(self.encoding)))
            else:
                # convert both images and labels to lmdb
                img_path = osp.join(data_root, img_name)
                if not osp.exists(img_path):
                    print('%s does not exist' % img_path)
                    continue
                with open(img_path, 'rb') as f:
                    image_bin = f.read()
                if self.verify:
                    try:
                        if not self.check_image_is_valid(image_bin):
                            print('%s is not a valid image' % img_path)
                            continue
                    except Exception:
                        print('error occurred at ', img_name)
                image_key = 'image-%09d'.encode(self.encoding) % cnt
                cache.append((image_key, image_bin))
                cache.append((label_key, text.encode(self.encoding)))
            
            if cnt % self.batch_size == 0:
                self.write_cache(env, cache)
                cache = []
                print('Written %d / %d' % (cnt, n_samples))
            cnt += 1
        n_samples = cnt - 1
        cache.append(
            ('num-samples'.encode(self.encoding), str(n_samples).encode(self.encoding)))
        self.write_cache(env, cache)
        print('Created lmdb dataset with %d samples' % n_samples)