# Copyright (c) OpenMMLab. All rights reserved.
import re
from typing import Dict, Tuple

import yaml

from ..data_preparer import DATA_PARSERS
from .base import BaseParser


@DATA_PARSERS.register_module()
class TotaltextTextDetAnnParser(BaseParser):
    """TotalText Text Detection Parser.

    The original annotation format of this dataset is stored in txt files,
    which is formed as the following format:
        x: [[x1 x2 x3 ... xn]], y: [[y1 y2 y3 ... yn]],
        ornt: [u'c'], transcriptions: [u'transcription']

    Args:
        data_root (str): Path to the dataset root.
        ignore (str): The text of the ignored instances. Default: '#'.
        nproc (int): Number of processes to load the data. Default: 1.
    """

    def __init__(self, ignore: str = '#', nproc: int = 1) -> None:
        self.ignore = ignore
        super().__init__(nproc=nproc)

    def parse_file(self, img_path: str, ann_path: str) -> Dict:
        """Convert single annotation."""
        instances = list()
        for poly, text in self.loader(ann_path):
            instances.append(
                dict(poly=poly, text=text, ignore=text == self.ignore))

        return img_path, instances

    def loader(self, file_path: str) -> str:
        """The annotation of the totaltext dataset may be stored in multiple
        lines, this loader is designed for this special case.

        Args:
            file_path (str): Path to the txt file

        Yield:
            str: Complete annotation of the txt file
        """

        def parsing_line(line: str) -> Tuple:
            """Parsing a line of the annotation.

            Args:
                line (str): A line of the annotation.

            Returns:
                Tuple: A tuple of (polygon, transcription).
            """
            line = '{' + line.replace('[[', '[').replace(']]', ']') + '}'
            ann_dict = re.sub('([0-9]) +([0-9])', r'\1,\2', line)
            ann_dict = re.sub('([0-9]) +([ 0-9])', r'\1,\2', ann_dict)
            ann_dict = re.sub('([0-9]) -([0-9])', r'\1,-\2', ann_dict)
            ann_dict = ann_dict.replace("[u',']", "[u'#']")
            ann_dict = yaml.safe_load(ann_dict)

            # polygon
            xs, ys = ann_dict['x'], ann_dict['y']
            poly = []
            for x, y in zip(xs, ys):
                poly.append(x)
                poly.append(y)
            # text
            text = ann_dict['transcriptions']
            if len(text) == 0:
                text = '#'
            else:
                word = text[0]
                if len(text) > 1:
                    for ann_word in text[1:]:
                        word += ',' + ann_word
                text = str(eval(word))

            return poly, text

        with open(file_path, 'r') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if idx == 0:
                    tmp_line = line
                    continue
                if not line.startswith('x:'):
                    tmp_line += ' ' + line
                    continue
                complete_line = tmp_line
                tmp_line = line
                yield parsing_line(complete_line)

            if tmp_line != '':
                yield parsing_line(tmp_line)
