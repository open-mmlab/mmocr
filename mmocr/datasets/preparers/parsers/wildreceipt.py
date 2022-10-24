# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from typing import Dict, Tuple

from mmocr.utils import list_from_file
from ..data_preparer import DATA_PARSER
from .base import BaseParser


@DATA_PARSER.register_module()
class WildreceiptTextDetParser(BaseParser):
    """Wildreceipt Text Detection Parser.

    The original annotation format of this dataset is stored in txt files,
    which is formed as the following json line format:
        {"file_name": "xxx/xxx/xx/xxxx.jpeg",
        "height": 1200,
        "width": 1600,
        "annotations": [
            "box": [x1, y1, x2, y2, x3, y3, x4, y4],
            "text": "xxx",
            "label": 25,
        ]}
    """

    def __init__(self,
                 data_root: str,
                 ignore: int = 0,
                 nproc: int = 1) -> None:
        self.ignore = ignore
        super().__init__(data_root=data_root, nproc=nproc)

    def parse_files(self, files: Tuple, split: str) -> Dict:
        """Convert single annotation."""
        closeset_lines = list_from_file(files)
        samples = list()
        for line in closeset_lines:
            instances = list()
            line = json.loads(line)
            img_file = osp.join(self.data_root, line['file_name'])
            for anno in line['annotations']:
                poly = anno['box']
                text = anno['text']
                label = anno['label']
                instances.append(
                    dict(poly=poly, text=text, ignore=label == self.ignore))
            samples.append((img_file, instances))

        return samples


@DATA_PARSER.register_module()
class WildreceiptKIEParser(BaseParser):
    """Wildreceipt KIE Parser.

    The original annotation format of this dataset is stored in txt files,
    which is formed as the following json line format:
        {"file_name": "xxx/xxx/xx/xxxx.jpeg",
        "height": 1200,
        "width": 1600,
        "annotations": [
            "box": [x1, y1, x2, y2, x3, y3, x4, y4],
            "text": "xxx",
            "label": 25,
        ]}
    """

    def __init__(self,
                 data_root: str,
                 ignore: int = 0,
                 nproc: int = 1) -> None:
        self.ignore = ignore
        super().__init__(data_root=data_root, nproc=nproc)

    def parse_files(self, files: Tuple, split: str) -> Dict:
        """Convert single annotation."""
        closeset_lines = list_from_file(files)

        return closeset_lines
