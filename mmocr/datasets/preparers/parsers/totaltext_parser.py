# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple

from ..data_preparer import DATA_PARSER
from .base import BaseParser
from .loaders import totaltext_loader


@DATA_PARSER.register_module()
class TotaltextTextDetParser(BaseParser):
    """TotalText Text Detection Parser.

    The original annotation format of this dataset is stored in txt files,
    which is formed as the following format:
        x: [[x1 x2 x3 ... xn]], y: [[y1 y2 y3 ... yn]],
        ornt: [u'c'], transcriptions: [u'transcription']
    """

    def __init__(self,
                 data_root: str,
                 ignore: str = '#',
                 nproc: int = 1) -> None:
        self.ignore = ignore
        super().__init__(data_root=data_root, nproc=nproc)

    def parse_file(self, file: Tuple, split: str) -> Dict:
        """Convert single annotation."""
        img_file, txt_file = file
        instances = list()
        for poly, text in totaltext_loader(txt_file):
            instances.append(
                dict(poly=poly, text=text, ignore=text == self.ignore))

        return img_file, instances
