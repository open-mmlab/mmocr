# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
from mmdet.data_elements.mask import PolygonMasks, bitmap_to_polygon

from mmocr.datasets import MMDet2MMOCR, MMOCR2MMDet, Resize
from mmocr.utils import poly2shapely


class TestMMDet2MMOCR(unittest.TestCase):

    def setUp(self):
        img = np.zeros((15, 30, 3))
        img_shape = (15, 30)
        polygons = [
            np.array([10., 5., 20., 5., 20., 10., 10., 10.]),
            np.array([10., 5., 20., 5., 20., 10., 10., 10., 8., 7.])
        ]
        ignores = np.array([True, False])
        bboxes = np.array([[10., 5., 20., 10.], [0., 0., 10., 10.]])
        self.data_info_ocr = dict(
            img=img,
            gt_polygons=polygons,
            gt_bboxes=bboxes,
            img_shape=img_shape,
            gt_ignored=ignores)

        _polygons = [[polygon] for polygon in polygons]
        masks = PolygonMasks(_polygons, *img_shape)
        self.data_info_det_polygon = dict(
            img=img,
            gt_masks=masks,
            gt_bboxes=bboxes,
            gt_ignore_flags=ignores,
            img_shape=img_shape)

        masks = masks.to_bitmap()
        self.data_info_det_mask = dict(
            img=img,
            gt_masks=masks,
            gt_bboxes=bboxes,
            gt_ignore_flags=ignores,
            img_shape=img_shape)

    def test_ocr2det_polygonmasks(self):
        transform = MMOCR2MMDet()
        results = transform(self.data_info_ocr.copy())
        self.assertEqual(results['img'].shape, (15, 30, 3))
        self.assertEqual(results['img_shape'], (15, 30))
        self.assertTrue(
            np.allclose(results['gt_masks'].masks[0][0],
                        self.data_info_det_polygon['gt_masks'].masks[0][0]))
        self.assertTrue(
            np.allclose(results['gt_masks'].masks[0][0],
                        self.data_info_det_polygon['gt_masks'].masks[0][0]))
        self.assertTrue(
            np.allclose(results['gt_bboxes'],
                        self.data_info_det_polygon['gt_bboxes']))
        self.assertTrue(
            np.allclose(results['gt_ignore_flags'],
                        self.data_info_det_polygon['gt_ignore_flags']))

    def test_ocr2det_bitmapmasks(self):
        transform = MMOCR2MMDet(poly2mask=True)
        results = transform(self.data_info_ocr.copy())
        self.assertEqual(results['img'].shape, (15, 30, 3))
        self.assertEqual(results['img_shape'], (15, 30))
        self.assertTrue(
            poly2shapely(
                bitmap_to_polygon(
                    results['gt_masks'].masks[0])[0][0].flatten()).equals(
                        poly2shapely(
                            bitmap_to_polygon(
                                self.data_info_det_mask['gt_masks'].masks[0])
                            [0][0].flatten())))

        self.assertTrue(
            np.allclose(results['gt_bboxes'],
                        self.data_info_det_mask['gt_bboxes']))
        self.assertTrue(
            np.allclose(results['gt_ignore_flags'],
                        self.data_info_det_mask['gt_ignore_flags']))

    def test_det2ocr_polygonmasks(self):
        transform = MMDet2MMOCR()
        results = transform(self.data_info_det_polygon.copy())
        self.assertEqual(results['img'].shape, (15, 30, 3))
        self.assertEqual(results['img_shape'], (15, 30))
        self.assertTrue(
            np.allclose(results['gt_polygons'][0],
                        self.data_info_ocr['gt_polygons'][0]))
        self.assertTrue(
            np.allclose(results['gt_polygons'][1],
                        self.data_info_ocr['gt_polygons'][1]))
        self.assertTrue(
            np.allclose(results['gt_bboxes'], self.data_info_ocr['gt_bboxes']))
        self.assertTrue(
            np.allclose(results['gt_ignored'],
                        self.data_info_ocr['gt_ignored']))

    def test_det2ocr_bitmapmasks(self):
        transform = MMDet2MMOCR()
        results = transform(self.data_info_det_mask.copy())
        self.assertEqual(results['img'].shape, (15, 30, 3))
        self.assertEqual(results['img_shape'], (15, 30))
        self.assertTrue(
            np.allclose(results['gt_bboxes'], self.data_info_ocr['gt_bboxes']))
        self.assertTrue(
            np.allclose(results['gt_ignored'],
                        self.data_info_ocr['gt_ignored']))

    def test_ocr2det2ocr(self):
        from mmdet.datasets.transforms import Resize as MMDet_Resize
        t1 = MMOCR2MMDet()
        t2 = MMDet_Resize(scale=(60, 60))
        t3 = MMDet2MMOCR()
        t4 = Resize(scale=(30, 15))
        results = t4(t3(t2(t1(self.data_info_ocr.copy()))))
        self.assertEqual(results['img'].shape, (15, 30, 3))
        self.assertTrue(
            np.allclose(results['gt_polygons'][0],
                        self.data_info_ocr['gt_polygons'][0]))
        self.assertTrue(
            np.allclose(results['gt_polygons'][1],
                        self.data_info_ocr['gt_polygons'][1]))
        self.assertTrue(
            np.allclose(results['gt_bboxes'], self.data_info_ocr['gt_bboxes']))
        self.assertEqual(results['gt_ignored'].all(),
                         self.data_info_ocr['gt_ignored'].all())

    def test_repr_det2ocr(self):
        transform = MMDet2MMOCR()
        self.assertEqual(repr(transform), ('MMDet2MMOCR'))

    def test_repr_ocr2det(self):
        transform = MMOCR2MMDet(poly2mask=True)
        self.assertEqual(repr(transform), ('MMOCR2MMDet(poly2mask = True)'))
