# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Any, Dict, List

import mmengine

from mmocr.utils import list_to_file
from ..data_preparer import DATA_DUMPERS


class BaseDumper:

    def __init__(self, task: str, split: str, data_root: str) -> None:
        self.task = task
        self.split = split
        self.data_root = data_root

    def __call__(self, data: Any) -> None:
        self.dump(data)

    def dump(self, data: Any) -> None:
        raise NotImplementedError


@DATA_DUMPERS.register_module()
class JsonDumper(BaseDumper):

    def dump(self, data: Dict) -> None:
        """Dump data to json file.

        Args:
            data (Dict): Data to be dumped.
        """

        filename = f'{self.task}_{self.split}.json'
        dst_file = osp.join(self.data_root, filename)
        mmengine.dump(data, dst_file)


@DATA_DUMPERS.register_module()
class WildreceiptOpensetDumper(BaseDumper):

    def dump(self, data: List):
        """Dump data to txt file.

        Args:
            data (List): Data to be dumped.
        """

        filename = f'openset_{self.split}.txt'
        dst_file = osp.join(self.data_root, filename)
        list_to_file(dst_file, data)
