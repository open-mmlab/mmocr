# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from mmengine.structures import InstanceData
from mmocr.structures import TextDetDataSample


def create_dummy_textdet_inputs(input_shape: Sequence[int] = (1, 3, 300, 300),
                                num_items: Optional[Sequence[int]] = None
                                ) -> Dict[str, Any]:
    """Create dummy inputs to test text detectors.

    Args:
        input_shape (tuple(int)): 4-d shape of the input image. Defaults to
            (1, 3, 300, 300).
        num_items (list[int], optional): Number of bboxes to create for each
            image. If None, they will be randomly generated. Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary of demo inputs.
    """
    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)

    metainfo = dict(
        img_shape=(H, W, C),
        ori_shape=(H, W, C),
        pad_shape=(H, W, C),
        filename='test.jpg',
        scale_factor=(1, 1),
        flip=False)

    gt_masks = []
    gt_kernels = []
    gt_effective_mask = []

    data_samples = []

    for batch_idx in range(N):
        if num_items is None:
            num_boxes = rng.randint(1, 10)
        else:
            num_boxes = num_items[batch_idx]

        data_sample = TextDetDataSample(
            metainfo=metainfo, gt_instances=InstanceData())

        cx, cy, bw, bh = rng.rand(num_boxes, 4).T

        tl_x = ((cx * W) - (W * bw / 2)).clip(0, W)
        tl_y = ((cy * H) - (H * bh / 2)).clip(0, H)
        br_x = ((cx * W) + (W * bw / 2)).clip(0, W)
        br_y = ((cy * H) + (H * bh / 2)).clip(0, H)

        boxes = np.vstack([tl_x, tl_y, br_x, br_y]).T
        class_idxs = [0] * num_boxes

        data_sample.gt_instances.bboxes = torch.FloatTensor(boxes)
        data_sample.gt_instances.labels = torch.LongTensor(class_idxs)
        data_sample.gt_instances.ignored = torch.BoolTensor([False] *
                                                            num_boxes)
        data_samples.append(data_sample)

        # kernels = []
        # TODO: add support for multiple kernels (if necessary)
        # for _ in range(num_kernels):
        #     kernel = np.random.rand(H, W)
        #     kernels.append(kernel)
        gt_kernels.append(np.random.rand(H, W))
        gt_effective_mask.append(np.ones((H, W)))

    mask = np.random.randint(0, 2, (len(boxes), H, W), dtype=np.uint8)
    gt_masks.append(mask)

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'data_samples': data_samples,
        'gt_masks': gt_masks,
        'gt_kernels': gt_kernels,
        'gt_mask': gt_effective_mask,
        'gt_thr_mask': gt_effective_mask,
        'gt_text_mask': gt_effective_mask,
        'gt_center_region_mask': gt_effective_mask,
        'gt_radius_map': gt_kernels,
        'gt_sin_map': gt_kernels,
        'gt_cos_map': gt_kernels,
    }
    return mm_inputs


def create_dummy_dict_file(
    dict_file: str,
    chars: List[str] = list('0123456789abcdefghijklmnopqrstuvwxyz')
) -> None:  # NOQA
    """Create a dummy dictionary file.

    Args:
        dict_file (str): Path to the dummy dictionary file.
        chars (list[str]): List of characters in dictionary. Defaults to
            ``list('0123456789abcdefghijklmnopqrstuvwxyz')``.
    """
    with open(dict_file, 'w') as f:
        for char in chars:
            f.write(char + '\n')
