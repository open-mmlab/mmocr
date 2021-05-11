import unittest.mock as mock

import numpy as np
import torchvision.transforms as TF
from PIL import Image

import mmocr.datasets.pipelines.dbnet_transforms as transforms
from mmdet.core import BitmapMasks, PolygonMasks


def test_imgaug():
    args=[['Fliplr', 0.5], dict(cls='Affine', rotate=[-10, 10]), ['Resize', [0.5, 3.0]]]
    imgaug = transforms.ImgAug(args)
    img = np.random.rand(3,100,200)
    poly = np.array([[[0,0,50,0,50,50,0,50]],[[20,20,50,20,50,50,20,50]]])
    box = np.array([[0,0,50,50],[20,20,50,50]])
    results = dict(img=img,masks=poly, bboxes=box)
    results['mask_fields'] = ['masks']
    results['bbox_fields'] = ['bboxes']
    results = imgaug(results)
    assert np.allclose(results['bboxes'][0],results['masks'].masks[0][0][[0,1,4,5]])
    assert np.allclose(results['bboxes'][1],results['masks'].masks[1][0][[0,1,4,5]])

def test_eastrandomcrop():
    crop = transforms.EastRandomCrop(target_size=(60,60),max_tries=100)
    img = np.random.rand(3,100,200)
    poly = np.array([[[0,0,50,0,50,50,0,50]],[[20,20,50,20,50,50,20,50]]])
    box = np.array([[0,0,50,50],[20,20,50,50]])
    results = dict(img=img,gt_masks=poly, bboxes=box)
    results['mask_fields'] = ['gt_masks']
    results['bbox_fields'] = ['bboxes']
    results = crop(results)
    print(results['gt_masks'].masks)
    print(results['bboxes'])

    assert np.allclose(results['bboxes'][0],results['gt_masks'].masks[0][0][[0,2]].flatten())
