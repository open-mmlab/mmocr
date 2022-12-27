# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import imgaug
import imgaug.augmenters as iaa
import numpy as np
import torchvision.transforms as torchvision_transforms
from mmcv.transforms.base import BaseTransform
from PIL import Image

from mmocr.registry import TRANSFORMS
from mmocr.utils import poly2bbox


@TRANSFORMS.register_module()
class ImgAugWrapper(BaseTransform):
    """A wrapper around imgaug https://github.com/aleju/imgaug.

    Find available augmenters at
    https://imgaug.readthedocs.io/en/latest/source/overview_of_augmenters.html.

    Required Keys:

    - img
    - gt_polygons (optional for text recognition)
    - gt_bboxes (optional for text recognition)
    - gt_bboxes_labels (optional for text recognition)
    - gt_ignored (optional for text recognition)
    - gt_texts (optional)

    Modified Keys:

    - img
    - gt_polygons (optional for text recognition)
    - gt_bboxes (optional for text recognition)
    - gt_bboxes_labels (optional for text recognition)
    - gt_ignored (optional for text recognition)
    - img_shape (optional)
    - gt_texts (optional)

    Args:
        args (list[list or dict]], optional): The argumentation list. For
            details, please refer to imgaug document. Take
            args=[['Fliplr', 0.5], dict(cls='Affine', rotate=[-10, 10]),
            ['Resize', [0.5, 3.0]]] as an example. The args horizontally flip
            images with probability 0.5, followed by random rotation with
            angles in range [-10, 10], and resize with an independent scale in
            range [0.5, 3.0] for each side of images. Defaults to None.
        fix_poly_trans (dict): The transform configuration to fix invalid
            polygons. Set it to None if no fixing is needed.
            Defaults to dict(type='FixInvalidPolygon').
    """

    def __init__(
        self,
        args: Optional[List[Union[List, Dict]]] = None,
        fix_poly_trans: Optional[dict] = dict(type='FixInvalidPolygon')
    ) -> None:
        assert args is None or isinstance(args, list) and len(args) > 0
        if args is not None:
            for arg in args:
                assert isinstance(arg, (list, dict)), \
                    'args should be a list of list or dict'
        self.args = args
        self.augmenter = self._build_augmentation(args)
        self.fix_poly_trans = fix_poly_trans
        if fix_poly_trans is not None:
            self.fix = TRANSFORMS.build(fix_poly_trans)

    def transform(self, results: Dict) -> Dict:
        """Transform the image and annotation data.

        Args:
            results (dict): Result dict containing the data to transform.

        Returns:
            dict: The transformed data.
        """
        # img is bgr
        image = results['img']
        aug = None
        ori_shape = image.shape

        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            if not self._augment_annotations(aug, ori_shape, results):
                return None
            results['img'] = aug.augment_image(image)
            results['img_shape'] = (results['img'].shape[0],
                                    results['img'].shape[1])
        if getattr(self, 'fix', None) is not None:
            results = self.fix(results)
        return results

    def _augment_annotations(self, aug: imgaug.augmenters.meta.Augmenter,
                             ori_shape: Tuple[int,
                                              int], results: Dict) -> Dict:
        """Augment annotations following the pre-defined augmentation sequence.

        Args:
            aug (imgaug.augmenters.meta.Augmenter): The imgaug augmenter.
            ori_shape (tuple[int, int]): The ori_shape of the original image.
            results (dict): Result dict containing annotations to transform.

        Returns:
            bool: Whether the transformation has been successfully applied. If
            the transform results in empty polygon/bbox annotations, return
            False.
        """
        # Assume co-existence of `gt_polygons`, `gt_bboxes` and `gt_ignored`
        # for text detection
        if 'gt_polygons' in results:

            # augment polygons
            transformed_polygons, removed_poly_inds = self._augment_polygons(
                aug, ori_shape, results['gt_polygons'])
            if len(transformed_polygons) == 0:
                return False
            results['gt_polygons'] = transformed_polygons

            # remove instances that are no longer inside the augmented image
            results['gt_bboxes_labels'] = np.delete(
                results['gt_bboxes_labels'], removed_poly_inds, axis=0)
            results['gt_ignored'] = np.delete(
                results['gt_ignored'], removed_poly_inds, axis=0)
            # TODO: deal with gt_texts corresponding to clipped polygons
            if 'gt_texts' in results:
                results['gt_texts'] = [
                    text for i, text in enumerate(results['gt_texts'])
                    if i not in removed_poly_inds
                ]

            # Generate new bboxes
            bboxes = [poly2bbox(poly) for poly in transformed_polygons]
            results['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
            if len(bboxes) > 0:
                results['gt_bboxes'] = np.stack(bboxes)

        return True

    def _augment_polygons(self, aug: imgaug.augmenters.meta.Augmenter,
                          ori_shape: Tuple[int, int], polys: List[np.ndarray]
                          ) -> Tuple[List[np.ndarray], List[int]]:
        """Augment polygons.

        Args:
            aug (imgaug.augmenters.meta.Augmenter): The imgaug augmenter.
            ori_shape (tuple[int, int]): The shape of the original image.
            polys (list[np.ndarray]): The polygons to be augmented.

        Returns:
            tuple(list[np.ndarray], list[int]): The augmented polygons, and the
            indices of polygons removed as they are out of the augmented image.
        """
        imgaug_polys = []
        for poly in polys:
            poly = poly.reshape(-1, 2)
            imgaug_polys.append(imgaug.Polygon(poly))
        imgaug_polys = aug.augment_polygons(
            [imgaug.PolygonsOnImage(imgaug_polys, shape=ori_shape)])[0]

        new_polys = []
        removed_poly_inds = []
        for i, poly in enumerate(imgaug_polys.polygons):
            # Sometimes imgaug may produce some invalid polygons with no points
            if not poly.is_valid or poly.is_out_of_image(imgaug_polys.shape):
                removed_poly_inds.append(i)
                continue
            new_poly = []
            try:
                poly = poly.clip_out_of_image(imgaug_polys.shape)[0]
            except Exception as e:
                warnings.warn(f'Failed to clip polygon out of image: {e}')
            for point in poly:
                new_poly.append(np.array(point, dtype=np.float32))
            new_poly = np.array(new_poly, dtype=np.float32).flatten()
            # Under some conditions, imgaug can generate "polygon" with only
            # two points, which is not a valid polygon.
            if len(new_poly) <= 4:
                removed_poly_inds.append(i)
                continue
            new_polys.append(new_poly)

        return new_polys, removed_poly_inds

    def _build_augmentation(self, args, root=True):
        """Build ImgAugWrapper augmentations.

        Args:
            args (dict): Arguments to be passed to imgaug.
            root (bool): Whether it's building the root augmenter.

        Returns:
            imgaug.augmenters.meta.Augmenter: The built augmenter.
        """
        if args is None:
            return None
        if isinstance(args, (int, float, str)):
            return args
        if isinstance(args, list):
            if root:
                sequence = [
                    self._build_augmentation(value, root=False)
                    for value in args
                ]
                return iaa.Sequential(sequence)
            arg_list = [self._to_tuple_if_list(a) for a in args[1:]]
            return getattr(iaa, args[0])(*arg_list)
        if isinstance(args, dict):
            if 'cls' in args:
                cls = getattr(iaa, args['cls'])
                return cls(
                    **{
                        k: self._to_tuple_if_list(v)
                        for k, v in args.items() if not k == 'cls'
                    })
            else:
                return {
                    key: self._build_augmentation(value, root=False)
                    for key, value in args.items()
                }
        raise RuntimeError('unknown augmenter arg: ' + str(args))

    def _to_tuple_if_list(self, obj: Any) -> Any:
        """Convert an object into a tuple if it is a list."""
        if isinstance(obj, list):
            return tuple(obj)
        return obj

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(args = {self.args}, '
        repr_str += f'fix_poly_trans = {self.fix_poly_trans})'
        return repr_str


@TRANSFORMS.register_module()
class TorchVisionWrapper(BaseTransform):
    """A wrapper around torchvision transforms. It applies specific transform
    to ``img`` and updates ``height`` and ``width`` accordingly.

    Required Keys:

    - img (ndarray): The input image.

    Modified Keys:

    - img (ndarray): The modified image.
    - img_shape (tuple(int, int)): The shape of the image in (height, width).


    Warning:
        This transform only affects the image but not its associated
        annotations, such as word bounding boxes and polygons. Therefore,
        it may only be applicable to text recognition tasks.

    Args:
        op (str): The name of any transform class in
            :func:`torchvision.transforms`.
        **kwargs: Arguments that will be passed to initializer of torchvision
            transform.
    """

    def __init__(self, op: str, **kwargs) -> None:
        assert isinstance(op, str)
        obj_cls = getattr(torchvision_transforms, op)
        self.torchvision = obj_cls(**kwargs)
        self.op = op
        self.kwargs = kwargs

    def transform(self, results):
        """Transform the image.

        Args:
            results (dict): Result dict from the data loader.

        Returns:
            dict: Transformed results.
        """
        assert 'img' in results
        # BGR -> RGB
        img = results['img'][..., ::-1]
        img = Image.fromarray(img)
        img = self.torchvision(img)
        img = np.asarray(img)
        img = img[..., ::-1]
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(op = {self.op}'
        for k, v in self.kwargs.items():
            repr_str += f', {k} = {v}'
        repr_str += ')'
        return repr_str
