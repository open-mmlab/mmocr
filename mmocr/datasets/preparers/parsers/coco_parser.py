# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

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
                 split: str,
                 nproc: int = 1,
                 variant: str = 'standard') -> None:

        super().__init__(nproc=nproc, split=split)
        assert variant in ['standard', 'cocotext', 'textocr'], \
            f'variant {variant} is not supported'
        self.variant = variant

    def parse_files(self, img_dir: str, ann_path: str) -> List:
        """Parse single annotation."""
        samples = list()
        coco = COCO(ann_path)
        if self.variant == 'cocotext' or self.variant == 'textocr':
            # cocotext stores both 'train' and 'val' split in one annotation
            # file, and uses the 'set' field to distinguish them.
            if self.variant == 'cocotext':
                for img in coco.dataset['imgs']:
                    if self.split == coco.dataset['imgs'][img]['set']:
                        coco.imgs[img] = coco.dataset['imgs'][img]
            # textocr stores 'train' and 'val'split separately
            elif self.variant == 'textocr':
                coco.imgs = coco.dataset['imgs']
            # both cocotext and textocr stores the annotation ID in the
            # 'imgToAnns' field, so we need to convert it to the 'anns' field
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
            ann_ids = coco.get_ann_ids(img_ids=[img_id])
            if len(ann_ids) == 0:
                continue
            ann_ids = [str(ann_id) for ann_id in ann_ids]
            ann_info = coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)
            instances = list()
            for ann in ann_info:
                if self.variant == 'standard':
                    # standard coco format use 'segmentation' field to store
                    # the polygon and 'iscrowd' field to store the ignore flag,
                    # and the 'text' field to store the text content.
                    instances.append(
                        dict(
                            poly=ann['segmentation'][0],
                            text=ann.get('text', None),
                            ignore=ann.get('iscrowd', False)))
                elif self.variant == 'cocotext':
                    # cocotext use 'utf8_string' field to store the text and
                    # 'legibility' field to store the ignore flag, and the
                    # 'mask' field to store the polygon.
                    instances.append(
                        dict(
                            poly=ann['mask'],
                            text=ann.get('utf8_string', None),
                            ignore=ann['legibility'] == 'illegible'))
                elif self.variant == 'textocr':
                    # textocr use 'utf8_string' field to store the text and
                    # the 'points' field to store the polygon, '.' is used to
                    # represent the ignored text.
                    text = ann.get('utf8_string', None)
                    instances.append(
                        dict(
                            poly=ann['points'], text=text, ignore=text == '.'))
            samples.append((osp.join(img_dir,
                                     osp.basename(img_path)), instances))
        return samples
