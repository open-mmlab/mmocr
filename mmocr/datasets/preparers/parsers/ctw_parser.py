# Copyright (c) OpenMMLab. All rights reserved.
import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np

from mmocr.datasets.preparers.data_preparer import DATA_PARSERS
from mmocr.datasets.preparers.parsers.base import BaseParser
from mmocr.utils import list_from_file


@DATA_PARSERS.register_module()
class CTWAnnParser(BaseParser):
    """SCUT-CTW dataset parser.

    Args:
        data_root (str): Path to the dataset root.
        ignore (list(str)): The text of the ignored instances. Default: ['#'].
        nproc (int): Number of processes to load the data. Default: 1.
    """

    def __init__(self,
                 data_root: str,
                 ignore: List[str] = ['#'],
                 nproc: int = 1) -> None:
        self.ignore = ignore
        super().__init__(data_root=data_root, nproc=nproc)

    def parse_file(self, file: Tuple, split: str) -> Tuple:
        """Parse single annotation.

        Args:
            file (tuple): Tuple of (img_file, json_file).
            split (str): Split of the file. For train split, xml file will be
                used. For test split, txt file will be used.

        Returns:
            tuple: Tuple of (img_file, instances).
        """
        img_dir, anno_dir = file
        if split == 'train':
            instances = self.load_xml_info(anno_dir)
        elif split == 'test':
            instances = self.load_txt_info(anno_dir)
        return img_dir, instances

    def load_txt_info(self, anno_dir: str) -> List:
        """Load the annotation of the SCUT-CTW dataset (test split).
        Args:
            anno_dir (str): Path to the annotation file.

        Returns:
            list[Dict]: List of instances.
        """
        instances = list()
        for line in list_from_file(anno_dir):
            # each line has one ploygen (n vetices), and one text.
            # e.g., 695,885,866,888,867,1146,696,1143,####Latin 9
            line = line.strip()
            strs = line.split(',')
            assert strs[28][0] == '#'
            xy = [int(x) for x in strs[0:28]]
            assert len(xy) == 28
            poly = np.array(xy).reshape(-1).tolist()
            text = strs[28][4:]
            instances.append(
                dict(poly=poly, text=text, ignore=text in self.ignore))
        return instances

    def load_xml_info(self, anno_dir: str) -> List:
        """Load the annotation of the SCUT-CTW dataset (train split).
        Args:
            anno_dir (str): Path to the annotation file.

        Returns:
            list[Dict]: List of instances.
        """
        obj = ET.parse(anno_dir)
        instances = list()
        for image in obj.getroot():  # image
            for box in image:  # image
                text = box[0].text
                segs = box[1].text
                pts = segs.strip().split(',')
                pts = [int(x) for x in pts]
                assert len(pts) == 28
                poly = np.array(pts).reshape(-1).tolist()
                instances.append(dict(poly=poly, text=text, ignore=0))
        return instances
