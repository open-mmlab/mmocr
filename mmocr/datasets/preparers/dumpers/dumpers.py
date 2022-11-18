# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, List, Optional

import mmengine

from mmocr.utils import list_to_file
from ..data_preparer import DATA_DUMPERS


@DATA_DUMPERS.register_module()
class JsonDumper:

    def __init__(self, task: str, dataset_name: str) -> None:
        self.task = task
        self.dataset_name = dataset_name

    def dump(self, data: Dict, data_root: str, split: str) -> str:
        """Dump data to json file.

        Args:
            data (Dict): Data to be dumped.
            data_root (str): Root directory of data.
            split (str): Split of data.
            cfg_path (str): Path to configs. Defaults to 'configs/'.

        Returns:
            str: String of dataset config.

        Examples:
        The returned dataset config
        >>> icdar2015_textrecog_train = dict(
        >>>     type='OCRDataset',
        >>>     data_root=ic15_rec_data_root,
        >>>     ann_file='textrecog_train.json',
        >>>     test_mode=False,
        >>>     pipeline=None)
        """

        dst_file = osp.join(data_root, f'{self.task}_{split}.json')
        mmengine.dump(data, dst_file)


@DATA_DUMPERS.register_module()
class WildreceiptOpensetDumper:

    def __init__(self, task: str, dataset_name: Optional[str] = None) -> None:
        self.task = task

    def dump(self, data: List, data_root: str, split: str) -> str:
        """Dump data to txt file.

        Args:
            data (List): Data to be dumped.
            data_root (str): Root directory of data.
            split (str): Split of data.
        """

        list_to_file(osp.join(data_root, f'openset_{split}.txt'), data)

        return None
