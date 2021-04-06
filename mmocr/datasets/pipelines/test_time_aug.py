import mmcv
import numpy as np

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.compose import Compose


@PIPELINES.register_module()
class MultiRotateAugOCR:
    """Test-time augmentation with multiple rotations in the case that
    img_height > img_width.

    An example configuration is as follows:

    .. code-block::

        rotate_degrees=[0, 90, 270],
        transforms=[
            dict(
                type='ResizeOCR',
                height=32,
                min_width=32,
                max_width=160,
                keep_aspect_ratio=True),
            dict(type='ToTensorOCR'),
            dict(type='NormalizeOCR', **img_norm_cfg),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'valid_ratio'
                ]),
        ]

    After MultiRotateAugOCR with above configuration, the results are wrapped
    into lists of the same length as follows:

    .. code-block::

        dict(
            img=[...],
            img_shape=[...]
            ...
        )

    Args:
        transforms (list[dict]): Transformation applied for each augmentation.
        rotate_degrees (list[int] | None): Degrees of anti-clockwise rotation.
        force_rotate (bool): If True, rotate image by 'rotate_degrees'
            while ignore image aspect ratio.
    """

    def __init__(self, transforms, rotate_degrees=None, force_rotate=False):
        self.transforms = Compose(transforms)
        self.force_rotate = force_rotate
        if rotate_degrees is not None:
            self.rotate_degrees = rotate_degrees if isinstance(
                rotate_degrees, list) else [rotate_degrees]
            assert mmcv.is_list_of(self.rotate_degrees, int)
            for degree in self.rotate_degrees:
                assert 0 <= degree < 360
                assert degree % 90 == 0
            if 0 not in self.rotate_degrees:
                self.rotate_degrees.append(0)
        else:
            self.rotate_degrees = [0]

    def __call__(self, results):
        """Call function to apply test time augment transformation to results.

        Args:
            results (dict): Result dict contains the data to be transformed.

        Returns:
           dict[str: list]: The augmented data, where each value is wrapped
               into a list.
        """
        img_shape = results['img_shape']
        ori_height, ori_width = img_shape[:2]
        if not self.force_rotate and ori_height <= ori_width:
            rotate_degrees = [0]
        else:
            rotate_degrees = self.rotate_degrees
        aug_data = []
        for degree in set(rotate_degrees):
            _results = results.copy()
            if degree == 0:
                pass
            elif degree == 90:
                _results['img'] = np.rot90(_results['img'], 1)
            elif degree == 180:
                _results['img'] = np.rot90(_results['img'], 2)
            elif degree == 270:
                _results['img'] = np.rot90(_results['img'], 3)
            data = self.transforms(_results)
            aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'rotate_degrees={self.rotate_degrees})'
        return repr_str
