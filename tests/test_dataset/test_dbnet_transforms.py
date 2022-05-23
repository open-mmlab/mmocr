# Copyright (c) OpenMMLab. All rights reserved.
import imgaug
import numpy as np
from shapely.geometry import Polygon

import mmocr.datasets.pipelines.dbnet_transforms as transforms


def test_imgaug():
    args = [dict(cls='Affine', translate_px=dict(x=-10, y=-10))]
    imgaug_transform = transforms.ImgAug(args, clip_invalid_ploys=False)
    img = np.random.rand(100, 200, 3)
    poly = np.array([[[0, 0, 50, 0, 50, 50, 0, 50]],
                     [[20, 20, 50, 20, 50, 50, 20, 50]]])
    box = np.array([[0, 0, 50, 50], [20, 20, 50, 50]])
    results = dict(img=img, masks=poly, bboxes=box)
    results['mask_fields'] = ['masks']
    results['bbox_fields'] = ['bboxes']
    results = imgaug_transform(results)
    for i in range(2):
        mask = results['masks'].masks[i][0]
        poly = imgaug.augmentables.polys.Polygon(mask.reshape(-1, 2))
        box = poly.to_bounding_box().clip_out_of_image(results['img_shape'])
        assert box.coords_almost_equals(results['bboxes'][i].reshape(-1, 2))

    args = [dict(cls='Affine', translate_px=dict(x=-10, y=-10))]
    imgaug_transform = transforms.ImgAug(args, clip_invalid_ploys=True)
    img = np.random.rand(100, 200, 3)
    poly = np.array([[[0, 0, 50, 0, 50, 50, 0, 50]],
                     [[20, 20, 50, 20, 50, 50, 20, 50]]])
    box = np.array([[0, 0, 50, 50], [20, 20, 50, 50]])
    poly_target = np.array([[[0, 0, 40, 0, 40, 40, 0, 40]],
                            [[10, 10, 40, 10, 40, 40, 10, 40]]])
    box_target = np.array([[0, 0, 40, 40], [10, 10, 40, 40]])
    results = dict(img=img, masks=poly, bboxes=box)
    results['mask_fields'] = ['masks']
    results['bbox_fields'] = ['bboxes']
    results = imgaug_transform(results)
    assert np.allclose(results['bboxes'], box_target)
    for i in range(2):
        poly1 = Polygon(results['masks'].masks[i][0].reshape(-1, 2))
        poly2 = Polygon(poly_target[i].reshape(-1, 2))
        assert poly1.equals(poly2)
        assert np.allclose(results['bboxes'][i], box_target[i])


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
