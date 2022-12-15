# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple

from mmdet.datasets.api_wrappers import COCO

from mmocr.datasets.preparers.data_preparer import DATA_PARSERS
from mmocr.datasets.preparers.parsers.base import BaseParser


@DATA_PARSERS.register_module()
class COCOTextDetAnnParser(BaseParser):
    """COCO Text Detection Parser.

    Args:
        data_root (str): The root path of the dataset. Defaults to None.
        nproc (int): The number of processes to parse the annotation. Defaults
            to 1.
    """

    def __init__(self, data_root: str = None, nproc: int = 1) -> None:

        super().__init__(nproc=nproc, data_root=data_root)

    def parse_files(self, files: Tuple, split: str = None) -> Dict:
        """Parse single annotation."""
        samples = list()
        coco = COCO(files)
        img_ids = coco.get_img_ids()

        total_ann_ids = []
        for img_id in img_ids:
            img_info = coco.load_imgs([img_id])[0]
            img_info['img_id'] = img_id
            img_path = img_info['file_name']
            ann_ids = coco.get_ann_ids(img_ids=[img_id])
            ann_info = coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)
            instances = list()
            for ann in ann_info:
                instances.append(
                    dict(
                        poly=ann['segmentation'][0],
                        text=ann.get('text', None),
                        ignore=ann.get('iscrowd', False)))
            samples.append((img_path, instances))
        return samples
