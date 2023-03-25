# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from typing import List

from mmocr.registry import DATA_PARSERS
from .base import BaseParser


@DATA_PARSERS.register_module()
class XFUNDSERAnnParser(BaseParser):
    """XFUND Semantic Entity Recognition Annotation Parser. See
    dataset_zoo/xfund/xx/sample_anno.md for annotation example.

    Args:
        nproc (int): The number of processes to parse the annotation. Defaults
            to 1.
    """

    def parse_files(self, img_dir: str, ann_path: str) -> List:
        """Parse annotations."""
        assert isinstance(ann_path, str)
        instances = list()
        for img_fname, width, height, instance in self.loader(ann_path):
            instances.append(
                dict(
                    img_path=osp.join(img_dir, img_fname),
                    width=width,
                    height=height,
                    instances=instance))
        return instances

    def loader(self, file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
            for i in range(len(data['documents'])):
                img_fname = data['documents'][i]['img']['fname']
                width = data['documents'][i]['img']['width']
                height = data['documents'][i]['img']['height']
                cur_doc_texts, cur_doc_bboxes = [], []
                cur_doc_labels, cur_doc_words = [], []
                for j in range(len(data['documents'][i]['document'])):
                    cur_item = data['documents'][i]['document'][j]
                    cur_doc_texts.append(cur_item['text'])
                    cur_doc_bboxes.append(cur_item['box'])
                    cur_doc_labels.append(cur_item['label'])
                    cur_doc_words.append(cur_item['words'])
                instance = dict(
                    texts=cur_doc_texts,
                    bboxes=cur_doc_bboxes,
                    labels=cur_doc_labels,
                    words=cur_doc_words)
                yield img_fname, width, height, instance


@DATA_PARSERS.register_module()
class XFUNDREAnnParser(BaseParser):
    """XFUND Relation Extraction Annotation Parser. See
    dataset_zoo/xfund/xx/sample_anno.md for annotation example.

    Args:
        nproc (int): The number of processes to parse the annotation. Defaults
            to 1.
    """

    # TODO: 完成RE parser
    def __init__(self, split: str, nproc: int = 1) -> None:
        super().__init__(split, nproc)
