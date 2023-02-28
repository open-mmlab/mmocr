# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict

import mmengine

from ..data_preparer import DATA_DUMPERS
from .base import BaseDumper


@DATA_DUMPERS.register_module()
class JsonDumper(BaseDumper):
    """Dumper for json file."""

    def dump(self, data: Dict) -> None:
        """Dump data to json file.

        Args:
            data (Dict): Data to be dumped.
        """

        filename = f'{self.task}_{self.split}.json'
        dst_file = osp.join(self.data_root, filename)
        mmengine.dump(data, dst_file)
