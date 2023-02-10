# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import list_from_file

from mmocr.registry import DATASETS
from mmocr.utils.parsers import LineJsonParser
from mmocr.utils.polygon_utils import sort_vertex8


@DATASETS.register_module()
class WildReceiptDataset(BaseDataset):
    """WildReceipt Dataset for key information extraction. There are two files
    to be loaded: metainfo and annotation. The metainfo file contains the
    mapping between classes and labels. The annotation file contains the all
    necessary information about the image, such as bounding boxes, texts, and
    labels etc.

    The metainfo file is a text file with the following format:

    .. code-block:: none

        0 Ignore
        1 Store_name_value
        2 Store_name_key

    The annotation format is shown as follows.

    .. code-block:: json

        {
            "file_name": "a.jpeg",
            "height": 348,
            "width": 348,
            "annotations": [
                {
                    "box": [
                        114.0,
                        19.0,
                        230.0,
                        19.0,
                        230.0,
                        1.0,
                        114.0,
                        1.0
                    ],
                    "text": "CHOEUN",
                    "label": 1
                },
                {
                    "box": [
                        97.0,
                        35.0,
                        236.0,
                        35.0,
                        236.0,
                        19.0,
                        97.0,
                        19.0
                    ],
                    "text": "KOREANRESTAURANT",
                    "label": 2
                }
            ]
        }

    Args:
        directed (bool): Whether to use directed graph. Defaults to False.
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (str or dict, optional): Meta information for dataset, such as
            class information. If it's a string, it will be treated as a path
            to the class file from which the class information will be loaded.
            Defaults to None.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (dict, optional): Prefix for training data. Defaults to
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
    METAINFO = {
        'category': [{
            'id': '0',
            'name': 'Ignore'
        }, {
            'id': '1',
            'name': 'Store_name_value'
        }, {
            'id': '2',
            'name': 'Store_name_key'
        }, {
            'id': '3',
            'name': 'Store_addr_value'
        }, {
            'id': '4',
            'name': 'Store_addr_key'
        }, {
            'id': '5',
            'name': 'Tel_value'
        }, {
            'id': '6',
            'name': 'Tel_key'
        }, {
            'id': '7',
            'name': 'Date_value'
        }, {
            'id': '8',
            'name': 'Date_key'
        }, {
            'id': '9',
            'name': 'Time_value'
        }, {
            'id': '10',
            'name': 'Time_key'
        }, {
            'id': '11',
            'name': 'Prod_item_value'
        }, {
            'id': '12',
            'name': 'Prod_item_key'
        }, {
            'id': '13',
            'name': 'Prod_quantity_value'
        }, {
            'id': '14',
            'name': 'Prod_quantity_key'
        }, {
            'id': '15',
            'name': 'Prod_price_value'
        }, {
            'id': '16',
            'name': 'Prod_price_key'
        }, {
            'id': '17',
            'name': 'Subtotal_value'
        }, {
            'id': '18',
            'name': 'Subtotal_key'
        }, {
            'id': '19',
            'name': 'Tax_value'
        }, {
            'id': '20',
            'name': 'Tax_key'
        }, {
            'id': '21',
            'name': 'Tips_value'
        }, {
            'id': '22',
            'name': 'Tips_key'
        }, {
            'id': '23',
            'name': 'Total_value'
        }, {
            'id': '24',
            'name': 'Total_key'
        }, {
            'id': '25',
            'name': 'Others'
        }]
    }

    def __init__(self,
                 directed: bool = False,
                 ann_file: str = '',
                 metainfo: Optional[Union[dict, str]] = None,
                 data_root: str = '',
                 data_prefix: dict = dict(img_path=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = ...,
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):
        self.directed = directed
        super().__init__(ann_file, metainfo, data_root, data_prefix,
                         filter_cfg, indices, serialize_data, pipeline,
                         test_mode, lazy_init, max_refetch)
        self._metainfo['dataset_type'] = 'WildReceiptDataset'
        self._metainfo['task_name'] = 'KIE'

    @classmethod
    def _load_metainfo(cls, metainfo: Union[str, dict] = None) -> dict:
        """Collect meta information from path to the class list or the
        dictionary of meta.

        Args:
            metainfo (str or dict): Path to the class list, or a meta
            information dict. If ``metainfo`` contains existed filename, it
            will be parsed by ``list_from_file``.

        Returns:
            dict: Parsed meta information.
        """
        cls_metainfo = copy.deepcopy(cls.METAINFO)
        if isinstance(metainfo, str):
            cls_metainfo['category'] = []
            for line in list_from_file(metainfo):
                k, v = line.split()
                cls_metainfo['category'].append({'id': k, 'name': v})
            return cls_metainfo
        else:
            return super()._load_metainfo(metainfo)

    def load_data_list(self) -> List[dict]:
        """Load data list from annotation file.

        Returns:
            List[dict]: A list of annotation dict.
        """
        parser = LineJsonParser(
            keys=['file_name', 'height', 'width', 'annotations'])
        data_list = []
        for line in list_from_file(self.ann_file):
            data_info = parser(line)
            data_info = self.parse_data_info(data_info)
            data_list.append(data_info)
        return data_list

    def parse_data_info(self, raw_data_info: dict) -> dict:
        """Parse data info from raw data info.

        Args:
            raw_data_info (dict): Raw data info.

        Returns:
            dict: Parsed data info.

            - img_path (str): Path to the image.
            - img_shape (tuple(int, int)): Image shape in (H, W).
            - instances (list[dict]): A list of instances.
              - bbox (ndarray(dtype=np.float32)): Shape (4, ). Bounding box.
              - text (str): Annotation text.
              - edge_label (int): Edge label.
              - bbox_label (int): Bounding box label.
        """

        raw_data_info['img_path'] = raw_data_info['file_name']
        data_info = super().parse_data_info(raw_data_info)
        annotations = data_info['annotations']

        assert 'box' in annotations[0]
        assert 'text' in annotations[0]

        instances = []

        for ann in annotations:
            instance = {}
            bbox = np.array(sort_vertex8(ann['box']), dtype=np.int32)
            bbox = np.array([
                bbox[0::2].min(), bbox[1::2].min(), bbox[0::2].max(),
                bbox[1::2].max()
            ],
                            dtype=np.int32)

            instance['bbox'] = bbox
            instance['text'] = ann['text']
            instance['bbox_label'] = ann.get('label', 0)
            instance['edge_label'] = ann.get('edge', 0)
            instances.append(instance)

        return dict(
            instances=instances,
            img_path=data_info['img_path'],
            img_shape=(data_info['height'], data_info['width']))
