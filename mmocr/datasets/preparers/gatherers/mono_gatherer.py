# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Tuple

from ..data_preparer import DATA_GATHERERS
from .base import BaseGatherer


@DATA_GATHERERS.register_module()
class MonoGatherer(BaseGatherer):
    """Gather the dataset file. Specifically for the case that only one
    annotation file is needed. For example,

            img_001.jpg \
            img_002.jpg ---> train.json
            img_003.jpg /

    Args:
        ann_name (str, optional): The name of the annotation file. Defaults to
            None.
        img_postfix_dir (str, optional): The postfix of the image directory.
            The actual directory is the concatenation of img_root and
            img_postfix_dir.
        split (str, optional): Split of the datasets.
        img_root (str, optional): The directory of the images. Defaults to
            None.
        ann_root (str, optional): The directory of the annotation files.
            Defaults to None.
    """

    def __init__(self, ann_name: str, **kwargs) -> None:
        super().__init__(**kwargs)

        self.ann_name = ann_name

    def __call__(self) -> Tuple[str, str]:
        """
        Returns:
            tuple(str, str): The directory of the image and the path of
            annotation file.
        """

        return (self.img_dir, osp.join(self.ann_dir, self.ann_name))
