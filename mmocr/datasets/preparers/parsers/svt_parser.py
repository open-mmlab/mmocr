# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import xml.etree.ElementTree as ET
from typing import List, Tuple

from ..data_preparer import DATA_PARSERS
from .base import BaseParser


@DATA_PARSERS.register_module()
class SVTTextDetAnnParser(BaseParser):
    """SVT Text Detection Parser.

    Args:
        data_root (str): The root of the dataset. Defaults to None.
        nproc (int): The number of processes to parse the annotation. Defaults
            to 1.
    """

    def parse_files(self, img_dir: str, ann_path: str, split: str) -> List:
        """Parse annotations."""
        assert isinstance(ann_path, str)
        samples = list()
        for img_name, instance in self.loader(ann_path):
            samples.append((osp.join(img_dir,
                                     osp.basename(img_name)), instance))

        return samples

    def loader(self, file_path: str) -> Tuple[str, List]:
        """Load annotation from SVT xml format file. See annotation example in
        dataset_zoo/svt/sample_anno.md.

        Args:
            file_path (str): The path of the annotation file.

        Returns:
            Tuple[str, List]: The image name and the annotation list.

        Yields:
            Iterator[Tuple[str, List]]: The image name and the annotation list.
        """
        tree = ET.parse(file_path)
        root = tree.getroot()
        for image in root.findall('image'):
            image_name = image.find('imageName').text
            instances = list()
            for rectangle in image.find('taggedRectangles'):
                x = int(rectangle.get('x'))
                y = int(rectangle.get('y'))
                w = int(rectangle.get('width'))
                h = int(rectangle.get('height'))
                # The text annotation of this dataset is not case sensitive.
                # All of the texts were labeled as upper case. We convert them
                # to lower case for convenience.
                text = rectangle.find('tag').text.lower()
                instances.append(
                    dict(
                        poly=[x, y, x + w, y, x + w, y + h, x, y + h],
                        text=text,
                        ignore=False))
            yield image_name, instances
