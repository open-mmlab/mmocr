# Copyright (c) OpenMMLab. All rights reserved.
import json
from typing import Tuple

from mmocr.utils import bbox2poly
from ..data_preparer import DATA_PARSERS
from .base import BaseParser


@DATA_PARSERS.register_module()
class FUNSDTextDetAnnParser(BaseParser):
    """FUNSD Text Detection Annotation Parser. See
    dataset_zoo/funsd/sample_anno.md for annotation example.

    Args:
        nproc (int): The number of processes to parse the annotation. Defaults
            to 1.
    """

    def parse_file(self, file: Tuple) -> Tuple:
        """Parse single annotation."""
        img_file, json_file = file
        instances = list()
        for poly, text, ignore in self.loader(json_file):
            instances.append(dict(poly=poly, text=text, ignore=ignore))

        return img_file, instances

    def loader(self, file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
            for form in data['form']:
                for word in form['words']:
                    poly = bbox2poly(word['box']).tolist()
                    text = word['text']
                    ignore = len(text) == 0
                    yield poly, text, ignore
