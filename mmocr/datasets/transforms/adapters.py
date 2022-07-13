# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from mmcv.transforms.base import BaseTransform
from mmdet.core import PolygonMasks
from mmdet.core.mask.structures import bitmap_to_polygon

from mmocr.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MMDet2MMOCR(BaseTransform):
    """Convert transforms's data format from MMDet to MMOCR.

    Required Keys:

    - gt_masks (PolygonMasks | BitmapMasks) (optional)
    - gt_ignore_flags (np.bool) (optional)

    Added Keys:

    - gt_polygons (list[np.ndarray])
    - gt_ignored (np.ndarray)
    """

    def transform(self, results: Dict) -> Dict:
        """Convert MMDet's data format to MMOCR's data format.

        Args:
            results (Dict): Result dict containing the data to transform.

        Returns:
            (Dict): The transformed data.
        """
        # gt_masks -> gt_polygons
        if 'gt_masks' in results.keys():
            gt_polygons = []
            gt_masks = results.pop('gt_masks')
            if len(gt_masks) > 0:
                # PolygonMasks
                if isinstance(gt_masks[0], PolygonMasks):
                    gt_polygons = [mask[0] for mask in gt_masks.masks]
                # BitmapMasks
                else:
                    polygons = []
                    for mask in gt_masks.masks:
                        contours, _ = bitmap_to_polygon(mask)
                        polygons += [
                            contour.reshape(-1) for contour in contours
                        ]
                    # filter invalid polygons
                    gt_polygons = []
                    for polygon in polygons:
                        if len(polygon) < 6:
                            continue
                        gt_polygons.append(polygon)

            results['gt_polygons'] = gt_polygons
        # gt_ignore_flags -> gt_ignored
        if 'gt_ignore_flags' in results.keys():
            gt_ignored = results.pop('gt_ignore_flags')
            results['gt_ignored'] = gt_ignored

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class MMOCR2MMDet(BaseTransform):
    """Convert transforms's data format from MMOCR to MMDet.

    Required Keys:

    - img_shape
    - gt_polygons (List[ndarray]) (optional)
    - gt_ignored (np.bool) (optional)

    Added Keys:

    - gt_masks (PolygonMasks | BitmapMasks) (optional)
    - gt_ignore_flags (np.bool) (optional)

    Args:
        poly2mask (bool): Whether to convert mask to bitmap. Default: True.
    """

    def __init__(self, poly2mask: bool = False) -> None:
        self.poly2mask = poly2mask

    def transform(self, results: Dict) -> Dict:
        """Convert MMOCR's data format to MMDet's data format.

        Args:
            results (Dict): Result dict containing the data to transform.

        Returns:
            (Dict): The transformed data.
        """
        # gt_polygons -> gt_masks
        if 'gt_polygons' in results.keys():
            gt_polygons = results.pop('gt_polygons')
            gt_polygons = [[gt_polygon] for gt_polygon in gt_polygons]
            gt_masks = PolygonMasks(gt_polygons, *results['img_shape'])

            if self.poly2mask:
                gt_masks = gt_masks.to_bitmap()

            results['gt_masks'] = gt_masks
        # gt_ignore_flags -> gt_ignored
        if 'gt_ignored' in results.keys():
            gt_ignored = results.pop('gt_ignored')
            results['gt_ignore_flags'] = gt_ignored

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(poly2mask = {self.poly2mask})'
        return repr_str
