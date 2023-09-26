from typing import Dict, List, Optional, Tuple
import mmcv
import numpy as np
from mmcv.transforms.base import BaseTransform

from mmocr.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomPad(BaseTransform):
    """Only pad the image's width.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_shape

    Added Keys:

    - pad_shape
    - pad_fixed_size
    - pad_size_divisor
    - valid_ratio

    Args:
        width (int): Target width of padded image. Defaults to None.
        pad_cfg (dict): Config to construct the Resize transform. Refer to
            ``Pad`` for detail. Defaults to ``dict(type='Pad')``.
    """

    def __init__(self, input_size: List[int], random_padding: bool = True, fill=0, pad_cfg: dict = dict(type='Pad')) -> None:
        super().__init__()
        height, width = input_size
        assert isinstance(width, int)
        assert isinstance(height, int)
        self.width = width
        self.height = height
        self.random_padding = random_padding
        self.fill = fill
        self.pad_cfg = pad_cfg
        _pad_cfg = self.pad_cfg.copy()
        _pad_cfg.update(dict(size=0))
        self.pad = TRANSFORMS.build(_pad_cfg)

    def transform(self, results: Dict) -> Dict:
        """Call function to pad images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        ori_height, ori_width = results['img'].shape[:2]
        delta_width = self.width - ori_width
        delta_height = self.height - ori_height
        if self.random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )

        results['img'] = mmcv.impad(results['img'], padding=padding, pad_val=self.fill, padding_mode='constant')
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(width={self.width}, '
        repr_str += f'(height={self.height}, '
        repr_str += f'(random_padding={self.random_padding}, '
        repr_str += f'pad_cfg={self.pad_cfg})'
        return repr_str
