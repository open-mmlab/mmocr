from mmdet.core import BitmapMasks
from mmdet.datasets.builder import PIPELINES
from . import BaseTextDetTargets


@PIPELINES.register_module()
class PANetTargets(BaseTextDetTargets):
    """Generate the ground truths for PANet: Efficient and Accurate Arbitrary-
    Shaped Text Detection with Pixel Aggregation Network.

    [https://arxiv.org/abs/1908.05900]. This code is partially adapted from
    https://github.com/WenmuZhou/PAN.pytorch.

    Args:
        shrink_ratio (tuple[float]): The ratios for shrinking text instances.
        max_shrink (int): The maximum shrink distance.
    """

    def __init__(self, shrink_ratio=(1.0, 0.5), max_shrink=20):
        self.shrink_ratio = shrink_ratio
        self.max_shrink = max_shrink

    def generate_targets(self, results):
        """Generate the gt targets for PANet.

        Args:
            results (dict): The input result dictionary.

        Returns:
            results (dict): The output result dictionary.
        """

        assert isinstance(results, dict)

        polygon_masks = results['gt_masks'].masks
        polygon_masks_ignore = results['gt_masks_ignore'].masks

        h, w, _ = results['img_shape']
        gt_kernels = []
        for ratio in self.shrink_ratio:
            mask, _ = self.generate_kernels((h, w),
                                            polygon_masks,
                                            ratio,
                                            max_shrink=self.max_shrink,
                                            ignore_tags=None)
            gt_kernels.append(mask)
        gt_mask = self.generate_effective_mask((h, w), polygon_masks_ignore)

        results['mask_fields'].clear()  # rm gt_masks encoded by polygons
        if 'bbox_fields' in results:
            results['bbox_fields'].clear()
        results.pop('gt_labels', None)
        results.pop('gt_masks', None)
        results.pop('gt_bboxes', None)
        results.pop('gt_bboxes_ignore', None)

        mapping = {'gt_kernels': gt_kernels, 'gt_mask': gt_mask}
        for key, value in mapping.items():
            value = value if isinstance(value, list) else [value]
            results[key] = BitmapMasks(value, h, w)
            results['mask_fields'].append(key)

        return results
