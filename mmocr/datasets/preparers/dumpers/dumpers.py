# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

import mmengine

from mmocr.utils import list_to_file
from ..data_preparer import DATA_DUMPER


@DATA_DUMPER.register_module()
class JsonDumper:

    def __init__(self, task: str) -> None:
        self.task = task
        self.format = format

    def dump(self, data: List, data_root: str, split: str) -> None:
        dst_file = osp.join(data_root, f'{self.task}_{split}.json')
        mmengine.dump(data, dst_file)


@DATA_DUMPER.register_module()
class WildreceiptOpensetDumper:

    def __init__(self, task: str) -> None:
        self.task = task

    def dump(self, data: List, data_root: str, split: str) -> None:
        list_to_file(osp.join(data_root, f'openset_{split}.txt'), data)
