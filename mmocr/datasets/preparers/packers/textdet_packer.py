# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, List, Tuple

import mmcv

from mmocr.utils import bbox2poly, poly2bbox
from ..data_preparer import DATA_PACKERS
from .base import BasePacker


@DATA_PACKERS.register_module()
class TextDetPacker(BasePacker):
    """Text detection packer. It is used to pack the parsed annotation info to.

    .. code-block:: python

        {
            "metainfo":
                {
                    "dataset_type": "TextDetDataset",
                    "task_name": "textdet",
                    "category": [{"id": 0, "name": "text"}]
                },
            "data_list":
                [
                    {
                        "img_path": "test_img.jpg",
                        "height": 640,
                        "width": 640,
                        "instances":
                        [
                            {
                                "polygon": [0, 0, 0, 10, 10, 20, 20, 0],
                                "bbox": [0, 0, 10, 20],
                                "bbox_label": 0,
                                "ignore": False
                            },
                            // ...
                        ]
                    }
                ]
        }
    """

    def pack_instance(self, sample: Tuple, bbox_label: int = 0) -> Dict:
        """Pack the parsed annotation info to an MMOCR format instance.

        Args:
            sample (Tuple): A tuple of (img_file, instances).
               - img_path (str): Path to the image file.
               - instances (Sequence[Dict]): A list of converted annos. Each
                 element should be a dict with the following keys:

                 - 'poly' or 'box'
                 - 'ignore'
                 - 'bbox_label' (optional)
            split (str): The split of the instance.

        Returns:
            Dict: An MMOCR format instance.
        """

        img_path, instances = sample

        img = mmcv.imread(img_path)
        h, w = img.shape[:2]

        packed_instances = list()
        for instance in instances:
            poly = instance.get('poly', None)
            box = instance.get('box', None)
            assert box or poly
            packed_sample = dict(
                polygon=poly if poly else list(
                    bbox2poly(box).astype('float64')),
                bbox=box if box else list(poly2bbox(poly).astype('float64')),
                bbox_label=bbox_label,
                ignore=instance['ignore'])
            packed_instances.append(packed_sample)

        packed_instances = dict(
            instances=packed_instances,
            img_path=osp.relpath(img_path, self.data_root),
            height=h,
            width=w)

        return packed_instances

    def add_meta(self, sample: List) -> Dict:
        """Add meta information to the sample.

        Args:
            sample (List): A list of samples of the dataset.

        Returns:
            Dict: A dict contains the meta information and samples.
        """
        meta = {
            'metainfo': {
                'dataset_type': 'TextDetDataset',
                'task_name': 'textdet',
                'category': [{
                    'id': 0,
                    'name': 'text'
                }]
            },
            'data_list': sample
        }
        return meta
