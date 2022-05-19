# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Optional, Tuple, Union

import imgaug
import imgaug.augmenters as iaa
import numpy as np
from mmcv.transforms.base import BaseTransform

from mmocr.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ImgAug(BaseTransform):
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
    """

    def __init__(self, args: Optional[List[Union[List, Dict]]] = None) -> None:
        assert args is None or isinstance(args, list) and len(args) > 0
        if args is not None:
            for arg in args:
                assert isinstance(arg, (list, dict)), \
                    'args should be a list of list or dict'
        self.args = args
        self.augmenter = self._build_augmentation(args)

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
            results['img'] = aug.augment_image(image)
            results['img_shape'] = (results['img'].shape[0],
                                    results['img'].shape[1])

            self._augment_annotations(aug, ori_shape, results)

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
            dict: The transformed data.
        """
        # Assume co-existence of `gt_polygons`, `gt_bboxes` and `gt_ignored`
        # for text detection
        if 'gt_polygons' in results:

            # augment polygons
            results['gt_polygons'], removed_poly_inds = self._augment_polygons(
                aug, ori_shape, results['gt_polygons'])

            # remove instances that are no longer inside the augmented image
            results['gt_bboxes'] = np.delete(
                results['gt_bboxes'], removed_poly_inds, axis=0)
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

            # augment bboxes
            bboxes = self._augment_bboxes(aug, ori_shape, results['gt_bboxes'])
            results['gt_bboxes'] = np.zeros((0, 4))
            if len(bboxes) > 0:
                results['gt_bboxes'] = np.stack(bboxes)

        return results

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
            if poly.is_out_of_image(imgaug_polys.shape):
                removed_poly_inds.append(i)
                continue
            new_poly = []
            for point in poly.clip_out_of_image(imgaug_polys.shape)[0]:
                new_poly.append(np.array(point, dtype=np.float32))
            new_poly = np.array(new_poly, dtype=np.float32).flatten()
            new_polys.append(new_poly)

        return new_polys, removed_poly_inds

    def _augment_bboxes(self, aug: imgaug.augmenters.meta.Augmenter,
                        ori_shape: Tuple[int, int],
                        bboxes: np.ndarray) -> np.ndarray:
        """Augment bboxes.

        Args:
            aug (imgaug.augmenters.meta.Augmenter): The imgaug augmenter.
            ori_shape (tuple[int, int]): The shape of the original image.
            bboxes (np.ndarray): The bboxes to be augmented.

        Returns:
            np.ndarray: The augmented bboxes.
        """
        imgaug_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            imgaug_bboxes.append(
                imgaug.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
        new_imgaug_bboxes = aug.augment_bounding_boxes([
            imgaug.BoundingBoxesOnImage(imgaug_bboxes, shape=ori_shape)
        ])[0].clip_out_of_image()

        new_bboxes = []
        for box in new_imgaug_bboxes.bounding_boxes:
            new_bboxes.append(
                np.array([box.x1, box.y1, box.x2, box.y2], dtype=np.float32))

        return new_bboxes

    def _build_augmentation(self, args, root=True):
        """Build ImgAug augmentations.

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
        repr_str += f'(args = {self.args})'
        return repr_str
