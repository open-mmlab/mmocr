# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import random
import tempfile
from unittest import TestCase, mock

import mmcv
import mmengine
import numpy as np
import torch

from mmocr.apis.inferencers import MMOCRInferencer


class TestMMOCRInferencer(TestCase):

    def setUp(self):
        seed = 1
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

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

    @mock.patch('mmengine.infer.infer._load_checkpoint')
    def test_init(self, mock_load):
        mock_load.side_effect = lambda *x, **y: None
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

    @mock.patch('mmengine.infer.infer._load_checkpoint')
    def test_det(self, mock_load):
        mock_load.side_effect = lambda *x, **y: None
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

        # test save_vis and save_pred
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = inferencer(
                img_paths, out_dir=tmp_dir, save_vis=True, save_pred=True)
            for img_dir in ['img_1.jpg', 'img_2.jpg']:
                self.assertTrue(osp.exists(osp.join(tmp_dir, 'vis', img_dir)))
            for i, pred_dir in enumerate(['img_1.json', 'img_2.json']):
                dumped_res = mmengine.load(
                    osp.join(tmp_dir, 'preds', pred_dir))
                self.assert_predictions_equal(res['predictions'][i],
                                              dumped_res)

    @mock.patch('mmengine.infer.infer._load_checkpoint')
    def test_rec(self, mock_load):
        mock_load.side_effect = lambda *x, **y: None
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

        # test save_vis and save_pred
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = inferencer(
                img_paths, out_dir=tmp_dir, save_vis=True, save_pred=True)
            for img_dir in ['1036169.jpg', '1058891.jpg']:
                self.assertTrue(osp.exists(osp.join(tmp_dir, 'vis', img_dir)))
            for i, pred_dir in enumerate(['1036169.json', '1058891.json']):
                dumped_res = mmengine.load(
                    osp.join(tmp_dir, 'preds', pred_dir))
                self.assert_predictions_equal(res['predictions'][i],
                                              dumped_res)

    @mock.patch('mmengine.infer.infer._load_checkpoint')
    def test_det_rec(self, mock_load):
        mock_load.side_effect = lambda *x, **y: None
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

        # test save_vis and save_pred
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = inferencer(
                img_paths, out_dir=tmp_dir, save_vis=True, save_pred=True)
            for img_dir in ['img_1.jpg', 'img_2.jpg']:
                self.assertTrue(osp.exists(osp.join(tmp_dir, 'vis', img_dir)))
            for i, pred_dir in enumerate(['img_1.json', 'img_2.json']):
                dumped_res = mmengine.load(
                    osp.join(tmp_dir, 'preds', pred_dir))
                self.assert_predictions_equal(res['predictions'][i],
                                              dumped_res)

        # corner case: when the det model cannot detect any texts
        inferencer(np.zeros((100, 100, 3)), return_vis=True)

    @mock.patch('mmengine.infer.infer._load_checkpoint')
    def test_dec_rec_kie(self, mock_load):
        mock_load.side_effect = lambda *x, **y: None
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

        # test save_vis and save_pred
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = inferencer(
                img_paths, out_dir=tmp_dir, save_vis=True, save_pred=True)
            for img_dir in ['1.jpg', '2.jpg']:
                self.assertTrue(osp.exists(osp.join(tmp_dir, 'vis', img_dir)))
            for i, pred_dir in enumerate(['1.json', '2.json']):
                dumped_res = mmengine.load(
                    osp.join(tmp_dir, 'preds', pred_dir))
                self.assert_predictions_equal(res['predictions'][i],
                                              dumped_res)
