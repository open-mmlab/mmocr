# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import re
from typing import List, Optional, Tuple

from mmocr.utils import list_files
from ..data_preparer import DATA_GATHERERS
from .base import BaseGatherer


@DATA_GATHERERS.register_module()
class PairGatherer(BaseGatherer):
    """Gather the dataset files. Specifically for the paired annotations. That
    is to say, each image has a corresponding annotation file. For example,

            img_1.jpg <---> gt_img_1.txt
            img_2.jpg <---> gt_img_2.txt
            img_3.jpg <---> gt_img_3.txt

    Args:
        img_suffixes (List[str]): File suffixes that used for searching.
        rule (Sequence): The rule for pairing the files. The
                first element is the matching pattern for the file, and the
                second element is the replacement pattern, which should
                be a regular expression. For example, to map the image
                name img_1.jpg to the annotation name gt_img_1.txt,
                the rule is
                    [r'img_(\d+)\.([jJ][pP][gG])', r'gt_img_\1.txt'] # noqa: W605 E501

    Note: PairGatherer assumes that each split annotation file is in the
    correspond split directory. For example, all the train annotation files are
    in {ann_dir}/train.
    """

    def __init__(self,
                 img_suffixes: Optional[List[str]] = None,
                 rule: Optional[List[str]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.rule = rule
        self.img_suffixes = img_suffixes
        # ann_dir = {ann_root}/{ann_dir}/{split}
        self.ann_dir = osp.join(self.ann_dir, self.split)

    def __call__(self) -> Tuple[List[str], List[str]]:
        """tuple(list, list):

        - first element: list of image paths.
        - second element: list of annotation paths.
        """

        img_list = list()
        ann_list = list()
        for img_path in list_files(self.img_dir, self.img_suffixes):
            if not re.match(self.rule[0], osp.basename(img_path)):
                continue
            ann_name = re.sub(self.rule[0], self.rule[1],
                              osp.basename(img_path))
            ann_path = osp.join(self.ann_dir, ann_name)
            img_list.append(img_path)
            ann_list.append(ann_path)

        return img_list, ann_list
