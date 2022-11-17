# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, List

import mmengine

from mmocr.utils import list_to_file
from ..data_preparer import DATA_DUMPERS


@DATA_DUMPERS.register_module()
class JsonDumper:

    def __init__(self, task: str) -> None:
        self.task = task

    def dump(self, data: Dict, data_root: str, split: str) -> str:
        """Dump data to json file.

        Args:
            data (Dict): Data to be dumped.
            data_root (str): Root directory of data.
            split (str): Split of data.
            cfg_path (str): Path to configs. Defaults to 'configs/'.

        Returns:
            str: Path to dumped json file relative to data_root.
        """

        filename = f'{self.task}_{split}.json'
        dst_file = osp.join(data_root, filename)
        mmengine.dump(data, dst_file)

        return filename


@DATA_DUMPERS.register_module()
class WildreceiptOpensetDumper:

    def __init__(self, task: str) -> None:
        self.task = task

    def dump(self, data: List, data_root: str, split: str) -> str:
        """Dump data to txt file.

        Args:
            data (List): Data to be dumped.
            data_root (str): Root directory of data.
            split (str): Split of data.

        Returns:
            str: Path to dumped txt file relative to data_root.
        """

        filename = f'openset_{split}.txt'
        dst_file = osp.join(data_root, filename)
        list_to_file(dst_file, data)

        return filename
