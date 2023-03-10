# Copyright (c) OpenMMLab. All rights reserved.
import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np

from mmocr.datasets.preparers.data_preparer import DATA_PARSERS
from mmocr.datasets.preparers.parsers.base import BaseParser
from mmocr.utils import list_from_file


@DATA_PARSERS.register_module()
class CTW1500AnnParser(BaseParser):
    """SCUT-CTW1500 dataset parser.

    Args:
        ignore (str): The text of the ignored instances. Defaults to
            '###'.
    """

    def __init__(self, ignore: str = '###', **kwargs) -> None:
        self.ignore = ignore
        super().__init__(**kwargs)

    def parse_file(self, img_path: str, ann_path: str) -> Tuple:
        """Convert annotation for a single image.

        Args:
            img_path (str): The path of image.
            ann_path (str): The path of annotation.

        Returns:
            Tuple: A tuple of (img_path, instance).

            - img_path (str): The path of image file, which can be read
              directly by opencv.
            - instance: instance is a list of dict containing parsed
              annotations, which should contain the following keys:

              - 'poly' or 'box' (textdet or textspotting)
              - 'text' (textspotting or textrecog)
              - 'ignore' (all task)

        Examples:
            An example of returned values:
            >>> ('imgs/train/xxx.jpg',
            >>> dict(
            >>>    poly=[[[0, 1], [1, 1], [1, 0], [0, 0]]],
            >>>    text='hello',
            >>>    ignore=False)
            >>> )
        """

        if self.split == 'train':
            instances = self.load_xml_info(ann_path)
        elif self.split == 'test':
            instances = self.load_txt_info(ann_path)
        return img_path, instances

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
                dict(poly=poly, text=text, ignore=text == self.ignore))
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
