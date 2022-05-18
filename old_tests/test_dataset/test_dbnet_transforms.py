# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

import mmocr.datasets.pipelines.dbnet_transforms as transforms


def test_eastrandomcrop():
    crop = transforms.EastRandomCrop(target_size=(60, 60), max_tries=100)
    img = np.random.rand(3, 100, 200)
    poly = np.array([[[0, 0, 50, 0, 50, 50, 0, 50]],
                     [[20, 20, 50, 20, 50, 50, 20, 50]]])
    box = np.array([[0, 0, 50, 50], [20, 20, 50, 50]])
    results = dict(img=img, gt_masks=poly, bboxes=box)
    results['mask_fields'] = ['gt_masks']
    results['bbox_fields'] = ['bboxes']
    results = crop(results)
    assert np.allclose(results['bboxes'][0],
                       results['gt_masks'].masks[0][0][[0, 2]].flatten())
