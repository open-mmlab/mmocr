# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, List, Tuple

import mmcv
from mmengine import mkdir_or_exist

from mmocr.registry import DATA_PACKERS
from mmocr.utils import bbox2poly, crop_img, poly2bbox, warp_img
from .base import BasePacker


@DATA_PACKERS.register_module()
class TextRecogPacker(BasePacker):
    """Text recogntion packer. It is used to pack the parsed annotation info
    to:

    .. code-block:: python

        {
            "metainfo":
                {
                    "dataset_type": "TextRecogDataset",
                    "task_name": "textrecog",
                },
            "data_list":
                [
                    {
                        "img_path": "textrecog_imgs/train/test_img.jpg",
                        "instances":
                            [
                                {
                                    "text": "GRAND"
                                }
                            ]
                    }
                ]
        }
    """

    def pack_instance(self, sample: Tuple) -> Dict:
        """Pack the text info to a recognition instance.

        Args:
            samples (Tuple): A tuple of (img_name, text).
            split (str): The split of the instance.

        Returns:
            Dict: The packed instance.
        """

        img_name, text = sample
        # TODO: remove hard code
        packed_instance = dict(
            instances=[dict(text=text)],
            img_path=osp.join('textrecog_imgs', self.split,
                              osp.basename(img_name)))

        return packed_instance

    def add_meta(self, sample: List) -> Dict:
        """Add meta information to the sample.

        Args:
            sample (List): A list of samples of the dataset.

        Returns:
            Dict: A dict contains the meta information and samples.
        """
        meta = {
            'metainfo': {
                'dataset_type': 'TextRecogDataset',
                'task_name': 'textrecog'
            },
            'data_list': sample
        }
        return meta


@DATA_PACKERS.register_module()
class TextRecogCropPacker(TextRecogPacker):
    """Text recognition packer with image cropper. It is used to pack the
    parsed annotation info and crop out the word images from the full-size
    ones.

    Args:
        crop_with_warp (bool): Whether to crop the text from the original
            image using opencv warpPerspective.
        jitter (bool): (Applicable when crop_with_warp=True)
            Whether to jitter the box.
        jitter_ratio_x (float): (Applicable when crop_with_warp=True)
            Horizontal jitter ratio relative to the height.
        jitter_ratio_y (float): (Applicable when crop_with_warp=True)
            Vertical jitter ratio relative to the height.
        long_edge_pad_ratio (float): (Applicable when crop_with_warp=False)
            The ratio of padding the long edge of the cropped image.
            Defaults to 0.1.
        short_edge_pad_ratio (float): (Applicable when crop_with_warp=False)
            The ratio of padding the short edge of the cropped image.
            Defaults to 0.05.
    """

    def __init__(self,
                 crop_with_warp: bool = False,
                 jitter: bool = False,
                 jitter_ratio_x: float = 0.0,
                 jitter_ratio_y: float = 0.0,
                 long_edge_pad_ratio: float = 0.0,
                 short_edge_pad_ratio: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.crop_with_warp = crop_with_warp
        self.jitter = jitter
        self.jrx = jitter_ratio_x
        self.jry = jitter_ratio_y
        self.lepr = long_edge_pad_ratio
        self.sepr = short_edge_pad_ratio
        # Crop converter crops the images of textdet to patches
        self.cropped_img_dir = 'textrecog_imgs'
        self.crop_save_path = osp.join(self.data_root, self.cropped_img_dir)
        mkdir_or_exist(self.crop_save_path)
        mkdir_or_exist(osp.join(self.crop_save_path, self.split))

    def pack_instance(self, sample: Tuple) -> List:
        """Crop patches from image.

        Args:
            samples (Tuple): A tuple of (img_name, text).

        Return:
            List: The list of cropped patches.
        """

        def get_box(instance: Dict) -> List:
            if 'box' in instance:
                return bbox2poly(instance['box']).tolist()
            if 'poly' in instance:
                return bbox2poly(poly2bbox(instance['poly'])).tolist()

        def get_poly(instance: Dict) -> List:
            if 'poly' in instance:
                return instance['poly']
            if 'box' in instance:
                return bbox2poly(instance['box']).tolist()

        data_list = []
        img_path, instances = sample
        img = mmcv.imread(img_path)
        for i, instance in enumerate(instances):
            if instance['ignore']:
                continue
            if self.crop_with_warp:
                poly = get_poly(instance)
                patch = warp_img(img, poly, self.jitter, self.jrx, self.jry)
            else:
                box = get_box(instance)
                patch = crop_img(img, box, self.lepr, self.sepr)
            if patch.shape[0] == 0 or patch.shape[1] == 0:
                continue
            text = instance['text']
            patch_name = osp.splitext(
                osp.basename(img_path))[0] + f'_{i}' + osp.splitext(
                    osp.basename(img_path))[1]
            dst_path = osp.join(self.crop_save_path, self.split, patch_name)
            mmcv.imwrite(patch, dst_path)
            rec_instance = dict(
                instances=[dict(text=text)],
                img_path=osp.join(self.cropped_img_dir, self.split,
                                  patch_name))
            data_list.append(rec_instance)

        return data_list

    def add_meta(self, sample: List) -> Dict:
        # Since the TextRecogCropConverter packs all of the patches in a single
        # image into a list, we need to flatten the list.
        sample = [item for sublist in sample for item in sublist]
        return super().add_meta(sample)
