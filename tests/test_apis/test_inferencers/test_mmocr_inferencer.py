# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

import mmcv
import mmengine
import numpy as np

from mmocr.apis.inferencers import MMOCRInferencer


class TestMMOCRInferencer(TestCase):

    def assert_predictions_equal(self, pred1, pred2):
        if 'det_polygons' in pred1:
            self.assertTrue(
                np.allclose(pred1['det_polygons'], pred2['det_polygons'], 0.1))
        if 'det_scores' in pred1:
            self.assertTrue(
                np.allclose(pred1['det_scores'], pred2['det_scores'], 0.1))
        if 'rec_texts' in pred1:
            self.assertEqual(pred1['rec_texts'], pred2['rec_texts'])
        if 'rec_scores' in pred1:
            self.assertTrue(
                np.allclose(pred1['rec_scores'], pred2['rec_scores'], 0.1))
        if 'kie_labels' in pred1:
            self.assertEqual(pred1['kie_labels'], pred2['kie_labels'])
        if 'kie_scores' in pred1:
            self.assertTrue(
                np.allclose(pred1['kie_scores'], pred2['kie_scores'], 0.1))
        if 'kie_edge_scores' in pred1:
            self.assertTrue(
                np.allclose(pred1['kie_edge_scores'], pred2['kie_edge_scores'],
                            0.1))
        if 'kie_edge_labels' in pred1:
            self.assertEqual(pred1['kie_edge_labels'],
                             pred2['kie_edge_labels'])

    def test_init(self):
        MMOCRInferencer(det='dbnet_resnet18_fpnc_1200e_icdar2015')
        MMOCRInferencer(
            det='configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py',
            det_weights='https://download.openmmlab.com/mmocr/textdet/dbnet/'
            'dbnet_resnet18_fpnc_1200e_icdar2015/'
            'dbnet_resnet18_fpnc_1200e_icdar2015_20220825_221614-7c0e94f2.pth')
        MMOCRInferencer(rec='crnn_mini-vgg_5e_mj')
        with self.assertRaises(ValueError):
            MMOCRInferencer(kie='sdmgr')
        with self.assertRaises(ValueError):
            MMOCRInferencer(det='dummy')

    def test_det(self):
        inferencer = MMOCRInferencer(det='dbnet_resnet18_fpnc_1200e_icdar2015')
        img_path = 'tests/data/det_toy_dataset/imgs/test/img_1.jpg'
        res_img_path = inferencer(img_path, return_vis=True)

        img_paths = [
            'tests/data/det_toy_dataset/imgs/test/img_1.jpg',
            'tests/data/det_toy_dataset/imgs/test/img_2.jpg'
        ]
        res_img_paths = inferencer(img_paths, return_vis=True)
        self.assert_predictions_equal(res_img_path['predictions'][0],
                                      res_img_paths['predictions'][0])
        self.assertTrue(
            np.allclose(res_img_path['visualization'][0],
                        res_img_paths['visualization'][0]))

        img_ndarray = mmcv.imread(img_path)
        res_img_ndarray = inferencer(img_ndarray, return_vis=True)

        img_ndarrays = [mmcv.imread(p) for p in img_paths]
        res_img_ndarrays = inferencer(img_ndarrays, return_vis=True)
        self.assert_predictions_equal(res_img_ndarray['predictions'][0],
                                      res_img_ndarrays['predictions'][0])
        self.assertTrue(
            np.allclose(res_img_ndarray['visualization'][0],
                        res_img_ndarrays['visualization'][0]))
        # cross checking: ndarray <-> path
        self.assert_predictions_equal(res_img_ndarray['predictions'][0],
                                      res_img_path['predictions'][0])
        self.assertTrue(
            np.allclose(res_img_ndarray['visualization'][0],
                        res_img_path['visualization'][0]))

    def test_rec(self):
        inferencer = MMOCRInferencer(rec='crnn_mini-vgg_5e_mj')
        img_path = 'tests/data/rec_toy_dataset/imgs/1036169.jpg'
        res_img_path = inferencer(img_path, return_vis=True)

        img_paths = [
            'tests/data/rec_toy_dataset/imgs/1036169.jpg',
            'tests/data/rec_toy_dataset/imgs/1058891.jpg'
        ]
        res_img_paths = inferencer(img_paths, return_vis=True)
        self.assert_predictions_equal(res_img_path['predictions'][0],
                                      res_img_paths['predictions'][0])
        self.assertTrue(
            np.allclose(res_img_path['visualization'][0],
                        res_img_paths['visualization'][0]))
        # cross checking: ndarray <-> path
        img_ndarray = mmcv.imread(img_path)
        res_img_ndarray = inferencer(img_ndarray, return_vis=True)

        img_ndarrays = [mmcv.imread(p) for p in img_paths]
        res_img_ndarrays = inferencer(img_ndarrays, return_vis=True)
        self.assert_predictions_equal(res_img_ndarray['predictions'][0],
                                      res_img_ndarrays['predictions'][0])
        self.assertTrue(
            np.allclose(res_img_ndarray['visualization'][0],
                        res_img_ndarrays['visualization'][0]))
        self.assert_predictions_equal(res_img_ndarray['predictions'][0],
                                      res_img_path['predictions'][0])
        self.assertTrue(
            np.allclose(res_img_ndarray['visualization'][0],
                        res_img_path['visualization'][0]))

    def test_det_rec(self):
        inferencer = MMOCRInferencer(
            det='dbnet_resnet18_fpnc_1200e_icdar2015',
            rec='crnn_mini-vgg_5e_mj')
        img_path = 'tests/data/det_toy_dataset/imgs/test/img_1.jpg'
        res_img_path = inferencer(img_path, return_vis=True)

        img_paths = [
            'tests/data/det_toy_dataset/imgs/test/img_1.jpg',
            'tests/data/det_toy_dataset/imgs/test/img_2.jpg'
        ]
        res_img_paths = inferencer(img_paths, return_vis=True)
        self.assert_predictions_equal(res_img_path['predictions'][0],
                                      res_img_paths['predictions'][0])
        self.assertTrue(
            np.allclose(res_img_path['visualization'][0],
                        res_img_paths['visualization'][0]))

        img_ndarray = mmcv.imread(img_path)
        res_img_ndarray = inferencer(img_ndarray, return_vis=True)

        img_ndarrays = [mmcv.imread(p) for p in img_paths]
        res_img_ndarrays = inferencer(img_ndarrays, return_vis=True)
        self.assert_predictions_equal(res_img_ndarray['predictions'][0],
                                      res_img_ndarrays['predictions'][0])
        self.assertTrue(
            np.allclose(res_img_ndarray['visualization'][0],
                        res_img_ndarrays['visualization'][0]))
        # cross checking: ndarray <-> path
        self.assert_predictions_equal(res_img_ndarray['predictions'][0],
                                      res_img_path['predictions'][0])
        self.assertTrue(
            np.allclose(res_img_ndarray['visualization'][0],
                        res_img_path['visualization'][0]))

    def test_dec_rec_kie(self):
        inferencer = MMOCRInferencer(
            det='dbnet_resnet18_fpnc_1200e_icdar2015',
            rec='crnn_mini-vgg_5e_mj',
            kie='sdmgr_unet16_60e_wildreceipt')
        img_path = 'tests/data/kie_toy_dataset/wildreceipt/1.jpeg'
        res_img_path = inferencer(img_path, return_vis=True)

        img_paths = [
            'tests/data/kie_toy_dataset/wildreceipt/1.jpeg',
            'tests/data/kie_toy_dataset/wildreceipt/2.jpeg'
        ]
        res_img_paths = inferencer(img_paths, return_vis=True)
        self.assert_predictions_equal(res_img_path['predictions'][0],
                                      res_img_paths['predictions'][0])
        self.assertTrue(
            np.allclose(res_img_path['visualization'][0],
                        res_img_paths['visualization'][0]))

        img_ndarray = mmcv.imread(img_path)
        res_img_ndarray = inferencer(img_ndarray, return_vis=True)

        img_ndarrays = [mmcv.imread(p) for p in img_paths]
        res_img_ndarrays = inferencer(img_ndarrays, return_vis=True)

        self.assert_predictions_equal(res_img_ndarray['predictions'][0],
                                      res_img_ndarrays['predictions'][0])
        self.assertTrue(
            np.allclose(res_img_ndarray['visualization'][0],
                        res_img_ndarrays['visualization'][0]))
        # cross checking: ndarray <-> path
        self.assert_predictions_equal(res_img_ndarray['predictions'][0],
                                      res_img_path['predictions'][0])
        self.assertTrue(
            np.allclose(res_img_ndarray['visualization'][0],
                        res_img_path['visualization'][0]))

        # test visualization
        # img_out_dir
        with tempfile.TemporaryDirectory() as tmp_dir:
            inferencer(img_paths, img_out_dir=tmp_dir)
            for img_dir in ['00000006.jpg', '00000007.jpg']:
                self.assertTrue(osp.exists(osp.join(tmp_dir, img_dir)))

        # pred_out_file
        with tempfile.TemporaryDirectory() as tmp_dir:
            pred_out_file = osp.join(tmp_dir, 'tmp.pkl')
            res = inferencer(
                img_path, print_result=True, pred_out_file=pred_out_file)
            dumped_res = mmengine.load(pred_out_file)
            self.assert_predictions_equal(res['predictions'],
                                          dumped_res['predictions'])
