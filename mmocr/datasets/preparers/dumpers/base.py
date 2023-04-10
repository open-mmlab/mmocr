# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any


class BaseDumper:
    """Base class for data dumpers.

    Args:
        task (str): Task type. Options are 'textdet', 'textrecog',
            'textspotter', and 'kie'. It is usually set automatically and users
             do not need to set it manually in config file in most cases.
        split (str): It' s the partition of the datasets. Options are 'train',
            'val' or 'test'. It is usually set automatically and users do not
            need to set it manually in config file in most cases. Defaults to
            None.
        data_root (str): The root directory of the image and
            annotation. It is usually set automatically and users do not need
            to set it manually in config file in most cases. Defaults to None.
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
