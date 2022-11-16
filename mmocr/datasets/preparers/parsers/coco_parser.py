# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, Tuple

from mmdet.datasets.api_wrappers import COCO

from mmocr.datasets.preparers.data_preparer import DATA_PARSERS
from mmocr.datasets.preparers.parsers.base import BaseParser


@DATA_PARSERS.register_module()
class COCOTextDetAnnParser(BaseParser):
    """COCO-like Format Text Detection Parser.

    Args:
        data_root (str): The root path of the dataset. Defaults to None.
        nproc (int): The number of processes to parse the annotation. Defaults
            to 1.
        variant (str): Variant of COCO dataset, options are ['standard',
            'cocotext', 'textocr']. Defaults to 'standard'.
    """

    def __init__(self,
                 data_root: str = None,
                 nproc: int = 1,
                 variant: str = 'standard') -> None:

        super().__init__(nproc=nproc, data_root=data_root)
        assert variant in ['standard', 'cocotext', 'textocr'], \
            f'variant {variant} is not supported'
        self.variant = variant

    def parse_files(self, files: Tuple, split: str = None) -> Dict:
        """Parse single annotation."""
        samples = list()
        coco = COCO(files)
        if self.variant == 'cocotext':
            for img in coco.dataset['imgs']:
                if split == coco.dataset['imgs'][img]['set']:
                    coco.imgs[img] = coco.dataset['imgs'][img]
            for img in coco.dataset['imgToAnns']:
                ann_ids = coco.dataset['imgToAnns'][img]
                anns = [
                    coco.dataset['anns'][str(ann_id)] for ann_id in ann_ids
                ]
                coco.dataset['imgToAnns'][img] = anns
                coco.imgToAnns = coco.dataset['imgToAnns']
                coco.anns = coco.dataset['anns']
        img_ids = coco.get_img_ids()
        total_ann_ids = []
        for img_id in img_ids:
            img_info = coco.load_imgs([img_id])[0]
            img_info['img_id'] = img_id
            img_path = img_info['file_name']
            if self.data_root is not None:
                img_path = osp.join(self.data_root, img_path)
            ann_ids = coco.get_ann_ids(img_ids=[img_id])
            if len(ann_ids) == 0:
                continue
            ann_ids = [str(ann_id) for ann_id in ann_ids]
            ann_info = coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)
            instances = list()
            for ann in ann_info:
                if self.variant == 'cocotext':
                    instances.append(
                        dict(
                            poly=ann['mask'],
                            text=ann.get('utf8_string', None),
                            ignore=ann['legibility'] == 'illegible'))
                elif self.variant == 'standard':
                    instances.append(
                        dict(
                            poly=ann['segmentation'][0],
                            text=ann.get('text', None),
                            ignore=ann.get('iscrowd', False)))
            samples.append((img_path, instances))
        return samples
