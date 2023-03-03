# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

from mmocr.utils import list_to_file
from ..data_preparer import DATA_DUMPERS
from .base import BaseDumper


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
