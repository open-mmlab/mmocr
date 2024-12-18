# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

from mmocr.registry import DATA_PARSERS
from mmocr.utils import bbox2poly
from .base import BaseParser


@DATA_PARSERS.register_module()
class DetextDetAnnParser(BaseParser):
    """Detext Txt Format Text Detection Annotation Parser.

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
        remove_strs (List[str], Optional): Used to remove redundant strings in
            the transcription. Defaults to None.
        mode (str, optional): The mode of the box converter. Supported modes
            are 'xywh' and 'xyxy'. Defaults to None.
    """

    def __init__(self,
                 separator: str = ',',
                 ignore: str = '###',
                 format: str = 'x1,y1,x2,y2,x3,y3,x4,y4,trans',
                 encoding: str = 'utf-8',
                 remove_strs: Optional[List[str]] = None,
                 mode: str = None,
                 **kwargs) -> None:
        self.sep = separator
        self.format = format
        self.encoding = encoding
        self.ignore = ignore
        self.mode = mode
        self.remove_strs = remove_strs
        super().__init__(**kwargs)

    def parse_file(self, img_path: str, ann_path: str) -> Tuple:
        """Parse single annotation."""
        instances = list()
        for anno in self.loader(ann_path, self.sep, self.format,
                                self.encoding):
            anno = list(anno.values())
            if self.remove_strs is not None:
                for strs in self.remove_strs:
                    for i in range(len(anno)):
                        if strs in anno[i]:
                            anno[i] = anno[i].replace(strs, '')
            poly = list(map(float, anno[0:-1]))
            if self.mode is not None:
                poly = bbox2poly(poly, self.mode)
                poly = poly.tolist()
            text = anno[-1]
            instances.append(
                dict(poly=poly, text=text, ignore=text == self.ignore))

        return img_path, instances
