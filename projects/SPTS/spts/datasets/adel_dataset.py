# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from mmdet.datasets.coco import CocoDataset
from mmengine.dataset.base_dataset import Compose as _Compose
from shapely.errors import ShapelyError

from mmocr.datasets.transforms import FixInvalidPolygon
from mmocr.registry import DATASETS


class Compose(_Compose):
    """A compose variant that can automatically fix invalid polygons."""

    def __init__(self, transforms: Optional[Sequence[Union[dict, Callable]]]):
        super().__init__(transforms=transforms)
        self.fix = FixInvalidPolygon()

    def __call__(self, data: dict) -> Optional[dict]:
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        for t in self.transforms:
            backup_data = copy.deepcopy(data)
            try:
                data = t(data)
            except ShapelyError:
                data = self.fix(backup_data)
                data = t(data)
            # The transform will return None when it failed to load images or
            # cannot find suitable augmentation parameters to augment the data.
            # Here we simply return None if the transform returns None and the
            # dataset will handle it by randomly selecting another data sample.
            if data is None:
                return None
        return data


@DATASETS.register_module()
class AdelDataset(CocoDataset):
    """Dataset for text detection while ann_file in coco format.

    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (dict): Prefix for training data. Defaults to
            dict(img_path='').
        filter_cfg (dict, optional): Config for filter data. Defaults to None.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Defaults to None which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy. Defaults
            to True.
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Defaults to False.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Defaults to False.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Defaults to 1000.
    """
    METAINFO = {'classes': ('text', )}

    def __init__(self,
                 *args,
                 pipeline: List[Union[dict, Callable]] = [],
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pipeline = Compose(pipeline)

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information loaded from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        img_path = osp.join(self.data_prefix['img_path'],
                            img_info['file_name'])
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        instances = []
        for ann in ann_info:
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore'] = 1
            else:
                instance['ignore'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]
            # instance['polygon'] = bezier2poly(
            #     np.array(ann['bezier_pts'], dtype=np.float32))
            instance['beziers'] = np.array(ann['bezier_pts'], dtype=np.float32)
            instance['text'] = ann['rec']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info
