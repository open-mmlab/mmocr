# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List, Optional, Tuple, Union


class BaseGatherer:
    """Base class for gather.

    Note: Gather assumes that all the annotation file is in the same directory
    and all the image files are in the same directory.

    Args:
        split (str, optional): List of splits to gather. It' s the partition of
            the datasets. Options are 'train', 'val' or 'test'. Defaults to
            None.
        data_root (str, optional): The root directory of the image and
            annotation. Defaults to None.
        img_dir(str, optional): The directory of the images. Defaults to None.
        ann_dir (str, optional): The directory of the annotation files.
            Defaults to None.
    """

    def __init__(self,
                 split: Optional[str] = None,
                 data_root: Optional[str] = None,
                 ann_dir: Optional[str] = None,
                 img_dir: Optional[str] = None) -> None:
        self.split = split
        self.data_root = data_root
        self.ann_dir = osp.join(data_root, ann_dir)
        self.img_dir = osp.join(data_root, img_dir)

    def __call__(self) -> Union[Tuple[List[str], List[str]], Tuple[str, str]]:
        """The return value of the gatherer is a tuple of two lists or strings.

        The first element is the list of image paths or the directory of the
        images. The second element is the list of annotation paths or the path
        of the annotation file which contains all the annotations.
        """
        raise NotImplementedError
