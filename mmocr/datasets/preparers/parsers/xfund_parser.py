# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from typing import List

from mmocr.registry import DATA_PARSERS
from .base import BaseParser


@DATA_PARSERS.register_module()
class XFUNDAnnParser(BaseParser):
    """XFUND Semantic Entity Recognition and Relation Extraction Annotation
    Parser. See dataset_zoo/xfund/xx/sample_anno.md for annotation example.

    Args:
        nproc (int): The number of processes to parse the annotation. Defaults
            to 1.
    """

    def parse_files(self, img_dir: str, ann_path: str) -> List:
        """Parse annotations."""
        assert isinstance(ann_path, str)
        samples = list()
        for img_fname, instance in self.loader(ann_path):
            samples.append((osp.join(img_dir, img_fname), instance))
        return samples

    def loader(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for i in range(len(data['documents'])):
                img_fname = data['documents'][i]['img']['fname']
                instances = list()
                for j in range(len(data['documents'][i]['document'])):
                    cur_item = data['documents'][i]['document'][j]
                    instance = dict(
                        text=cur_item['text'],
                        box=cur_item['box'],
                        label=cur_item['label'],
                        words=cur_item['words'],
                        linking=cur_item['linking'],
                        id=cur_item['id'])
                    instances.append(instance)
                yield img_fname, instances
