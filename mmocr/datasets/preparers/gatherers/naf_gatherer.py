# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import os.path as osp
import shutil
from typing import List, Tuple

from ..data_preparer import DATA_GATHERERS
from .base import BaseGatherer


@DATA_GATHERERS.register_module()
class NAFGatherer(BaseGatherer):
    """Gather the dataset file from NAF dataset. Specifically for the case that
    there is a split file that contains the names of different splits. For
    example,

        img_001.jpg                           train: img_001.jpg
        img_002.jpg ---> split_file ---> test: img_002.jpg
        img_003.jpg                           val: img_003.jpg

    Args:
        split_file (str, optional): The name of the split file. Defaults to
            "data_split.json".
        temp_dir (str, optional): The directory of the temporary images.
            Defaults to "temp_images".
    """

    def __init__(self,
                 split_file='data_split.json',
                 temp_dir: str = 'temp_images',
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.temp_dir = temp_dir
        self.split_file = split_file

    def __call__(self) -> Tuple[List[str], List[str]]:
        """
        Returns:
            tuple(list[str], list[str]): The list of image paths and the list
            of annotation paths.
        """

        split_file = osp.join(self.data_root, self.split_file)
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        img_list = list()
        ann_list = list()
        # Rename the key
        split_data['val'] = split_data.pop('valid')
        if not osp.exists(self.img_dir):
            os.makedirs(self.img_dir)
        current_split_data = split_data[self.split]
        for groups in current_split_data:
            for img_name in current_split_data[groups]:
                src_img = osp.join(self.data_root, self.temp_dir, img_name)
                dst_img = osp.join(self.img_dir, img_name)
                if not osp.exists(src_img):
                    Warning(f'{src_img} does not exist!')
                    continue
                # move the image to the new path
                shutil.move(src_img, dst_img)
                ann = osp.join(self.ann_dir, img_name.replace('.jpg', '.json'))
                img_list.append(dst_img)
                ann_list.append(ann)
        return img_list, ann_list
