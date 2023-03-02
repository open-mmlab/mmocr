# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from typing import Dict

from mmocr.utils import list_from_file
from ..data_preparer import DATA_PARSERS
from .base import BaseParser


@DATA_PARSERS.register_module()
class WildreceiptTextDetAnnParser(BaseParser):
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

    Args:
        data_root (str): The root path of the dataset.
        ignore (int): The label to be ignored. Defaults to 0.
        nproc (int): The number of processes to parse the annotation. Defaults
            to 1.
    """

    def __init__(self, ignore: int = 0, **kwargs) -> None:
        self.ignore = ignore
        super().__init__(**kwargs)

    def parse_files(self, img_dir: str, ann_path) -> Dict:
        """Convert single annotation."""
        closeset_lines = list_from_file(ann_path)
        samples = list()
        for line in closeset_lines:
            instances = list()
            line = json.loads(line)
            img_file = osp.join(img_dir, osp.basename(line['file_name']))
            for anno in line['annotations']:
                poly = anno['box']
                text = anno['text']
                label = anno['label']
                instances.append(
                    dict(poly=poly, text=text, ignore=label == self.ignore))
            samples.append((img_file, instances))

        return samples


@DATA_PARSERS.register_module()
class WildreceiptKIEAnnParser(BaseParser):
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

    Args:
        ignore (int): The label to be ignored. Defaults to 0.
        nproc (int): The number of processes to parse the annotation. Defaults
            to 1.
    """

    def __init__(self, ignore: int = 0, **kwargs) -> None:
        self.ignore = ignore
        super().__init__(**kwargs)

    def parse_files(self, img_dir: str, ann_path: str) -> Dict:
        """Convert single annotation."""
        closeset_lines = list_from_file(ann_path)
        samples = list()
        for line in closeset_lines:
            json_line = json.loads(line)
            img_file = osp.join(img_dir, osp.basename(json_line['file_name']))
            json_line['file_name'] = img_file
            samples.append(json.dumps(json_line))

        return samples
