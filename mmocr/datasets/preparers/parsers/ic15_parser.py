# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

from ..data_preparer import DATA_PARSERS
from .base import BaseParser


@DATA_PARSERS.register_module()
class ICDARTxtTextDetAnnParser(BaseParser):
    """ICDAR2015 Text Detection Parser.

    The original annotation format of this dataset is stored in txt files,
    which is formed as the following format:
        x1, y1, x2, y2, x3, y3, x4, y4, transcription

    Args:
        separator (str): The separator between each element in a line. Defaults
            to ','.
        ignore (str): The text to be ignored. Defaults to '###'.
        format (str): The format of the annotation. Defaults to
            'x1,y1,x2,y2,x3,y3,x4,trans'.
        encoding (str): The encoding of the annotation file. Defaults to
            'utf-8-sig'.
        nproc (int): The number of processes to parse the annotation. Defaults
            to 1.
        remove_flag (List[str], Optional): Used to remove redundant strings in
            the transcription. Defaults to None.
    """

    def __init__(self,
                 separator: str = ',',
                 ignore: str = '###',
                 format: str = 'x1,y1,x2,y2,x3,y3,x4,y4,trans',
                 encoding: str = 'utf-8-sig',
                 nproc: int = 1,
                 remove_flag: Optional[List[str]] = None) -> None:
        self.sep = separator
        self.format = format
        self.encoding = encoding
        self.ignore = ignore
        self.remove_flag = remove_flag
        super().__init__(nproc=nproc)

    def parse_file(self, file: Tuple, split: str) -> Tuple:
        """Parse single annotation."""
        img_file, txt_file = file
        instances = list()
        for anno in self.loader(txt_file, self.sep, self.format,
                                self.encoding):
            anno = list(anno.values())
            poly = list(map(float, anno[0:-1]))
            text = anno[-1]
            if self.remove_flag is not None:
                for flag in self.remove_flag:
                    text = text.replace(flag, '')
            instances.append(
                dict(poly=poly, text=text, ignore=text == self.ignore))

        return img_file, instances


@DATA_PARSERS.register_module()
class ICDARTxtTextRecogAnnParser(BaseParser):
    """ICDAR2015 Text Detection Parser.

    The original annotation format of this dataset is stored in txt files,
    which is formed as the following format:
        img_path, transcription

    Args:
        separator (str): The separator between each element in a line. Defaults
            to ','.
        ignore (str): The text to be ignored. Defaults to '#'.
        format (str): The format of the annotation. Defaults to 'img, text'.
        encoding (str): The encoding of the annotation file. Defaults to
            'utf-8-sig'.
        nproc (int): The number of processes to parse the annotation. Defaults
            to 1.
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
        for anno in self.loader(
                file_path=files, format=self.format, encoding=self.encoding):
            text = anno['text'].strip().replace('"', '')
            samples.append((anno['img'], text))

        return samples
