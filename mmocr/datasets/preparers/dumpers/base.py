# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any


class BaseDumper:
    """Base class for data dumpers.

    Args:
        task (str): Task type. Options are 'textdet', 'textrecog',
            'textspotter', and 'kie'.
        split (str): Split type. Options are 'train', 'val', 'test'.
        data_root (str): Root directory of data.
    """

    def __init__(self, task: str, split: str, data_root: str) -> None:
        self.task = task
        self.split = split
        self.data_root = data_root

    def __call__(self, data: Any) -> None:
        """Call function.

        Args:
            data (Any): Data to be dumped.
        """
        self.dump(data)

    def dump(self, data: Any) -> None:
        raise NotImplementedError
