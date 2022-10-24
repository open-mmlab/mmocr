# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

from ..data_preparer import DATA_PARSER
from .base import BaseParser
from .loaders import txt_loader


@DATA_PARSER.register_module()
class ICDAR2015TextDetParser(BaseParser):
    """ICDAR2015 Text Detection Parser.

    The original annotation format of this dataset is stored in txt files,
    which is formed as the following format: x1, y1, x2, y2, x3, y3, x4, y4,
    transcription
    """

    def __init__(self,
                 separator: str = ',',
                 ignore: str = '###',
                 format: str = 'x1,y1,x2,y2,x3,y3,x4,y4,trans',
                 encoding: str = 'utf-8-sig',
                 nproc: int = 1) -> None:
        self.sep = separator
        self.format = format
        self.encoding = encoding
        self.ignore = ignore
        super().__init__(nproc=nproc)

    def parse_file(self, file: Tuple, split: str) -> Dict:
        """Parse single annotation."""
        img_file, txt_file = file
        instances = list()
        for anno in txt_loader(txt_file, self.sep, self.format, self.encoding):
            anno = list(anno.values())
            poly = list(map(float, anno[0:-1]))
            text = anno[-1]
            instances.append(
                dict(poly=poly, text=text, ignore=text == self.ignore))

        return img_file, instances


@DATA_PARSER.register_module()
class ICDAR2015TextRecogParser(BaseParser):
    """ICDAR2015 Text Detection Parser.

    The original annotation format of this dataset is stored in txt files,
    which is formed as the following format: img_path, transcription
    """

    def __init__(self,
                 separator: str = ',',
                 ignore: str = '#',
                 format: str = 'img,text',
                 encoding: str = 'utf-8-sig',
                 nproc: int = 1) -> None:
        self.sep = separator
        self.format = format
        self.encoding = encoding
        self.ignore = ignore
        super().__init__(nproc=nproc)

    def parse_files(self, files: str, split: str) -> List:
        """Parse annotations."""
        assert isinstance(files, str)
        samples = list()
        for anno in txt_loader(
                file_path=files, format=self.format, encoding=self.encoding):
            text = anno['text'].strip().replace('"', '')
            samples.append((anno['img'], text))

        return samples
