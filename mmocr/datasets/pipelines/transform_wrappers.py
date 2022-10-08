# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import random

import mmcv
import numpy as np
import torchvision.transforms as torchvision_transforms
from mmcv.utils import build_from_cfg
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Compose
from PIL import Image


@PIPELINES.register_module()
class OneOfWrapper:
    """Randomly select and apply one of the transforms, each with the equal
    chance.

    Warning:
        Different from albumentations, this wrapper only runs the selected
        transform, but doesn't guarantee the transform can always be applied to
        the input if the transform comes with a probability to run.

    Args:
        transforms (list[dict|callable]): Candidate transforms to be applied.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, list) or isinstance(transforms, tuple)
        assert len(transforms) > 0, 'Need at least one transform.'
        self.transforms = []
        for t in transforms:
            if isinstance(t, dict):
                self.transforms.append(build_from_cfg(t, PIPELINES))
            elif callable(t):
                self.transforms.append(t)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, results):
        return random.choice(self.transforms)(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms})'
        return repr_str


@PIPELINES.register_module()
class RandomWrapper:
    """Run a transform or a sequence of transforms with probability p.

    Args:
        transforms (list[dict|callable]): Transform(s) to be applied.
        p (int|float): Probability of running transform(s).
    """

    def __init__(self, transforms, p):
        assert 0 <= p <= 1
        self.transforms = Compose(transforms)
        self.p = p

    def __call__(self, results):
        return results if np.random.uniform() > self.p else self.transforms(
            results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'p={self.p})'
        return repr_str


@PIPELINES.register_module()
class TorchVisionWrapper:
    """A wrapper of torchvision transforms. It applies specific transform to
    ``img`` and updates ``img_shape`` accordingly.

    Warning:
        This transform only affects the image but not its associated
        annotations, such as word bounding boxes and polygon masks. Therefore,
        it may only be applicable to text recognition tasks.

    Args:
        op (str): The name of any transform class in
            :func:`torchvision.transforms`.
        **kwargs: Arguments that will be passed to initializer of torchvision
            transform.

    :Required Keys:
        - | ``img`` (ndarray): The input image.

    :Affected Keys:
        :Modified:
            - | ``img`` (ndarray): The modified image.
        :Added:
            - | ``img_shape`` (tuple(int)): Size of the modified image.
    """

    def __init__(self, op, **kwargs):
        assert type(op) is str

        if mmcv.is_str(op):
            obj_cls = getattr(torchvision_transforms, op)
        elif inspect.isclass(op):
            obj_cls = op
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(type)}')
        self.transform = obj_cls(**kwargs)
        self.kwargs = kwargs

    def __call__(self, results):
        assert 'img' in results
        # BGR -> RGB
        img = results['img'][..., ::-1]
        img = Image.fromarray(img)
        img = self.transform(img)
        img = np.asarray(img)
        img = img[..., ::-1]
        results['img'] = img
        results['img_shape'] = img.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transform={self.transform})'
        return repr_str
