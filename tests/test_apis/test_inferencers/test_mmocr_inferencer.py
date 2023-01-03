# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

import mmcv
import mmengine
import numpy as np

from mmocr.apis.inferencers import MMOCRInferencer
from mmocr.utils.check_argument import is_type_list
from mmocr.utils.typing_utils import TextDetDataSample


class TestTextDetinferencer(TestCase):

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

    # def test_init(self):
    #     # init from metafile
    #     TextDetInferencer('')
    #     # init from cfg
    #     TextDetInferencer(
    #         'configs/textdet/dbnet/'
    #         'dbnet_resnet18_fpnc_1200e_icdar2015.py',
    #         'https://download.openmmlab.com/mmocr/textdet/dbnet/'
    #         'dbnet_resnet18_fpnc_1200e_icdar2015/'
    #         'dbnet_resnet18_fpnc_1200e_icdar2015_20220825_221614-7c0e94f2.pth')

    # def assert_predictions_equal(self, preds1, preds2):
    #     for pred1, pred2 in zip(preds1, preds2):
    #         self.assertTrue(
    #             np.allclose(pred1['polygons'], pred2['polygons'], 0.1))
    #         self.assertTrue(np.allclose(pred1['scores'], pred2['scores'], 0.1))

    # def test_call(self):
    #     # single img
    #     img_path = 'tests/data/det_toy_dataset/imgs/test/img_1.jpg'
    #     res_path = self.inferencer(img_path, return_vis=True)
    #     # ndarray
    #     img = mmcv.imread(img_path)
    #     res_ndarray = self.inferencer(img, return_vis=True)
    #     self.assert_predictions_equal(res_path['predictions'],
    #                                   res_ndarray['predictions'])
    #     self.assertIn('visualization', res_path)
    #     self.assertIn('visualization', res_ndarray)

    #     # multiple images
    #     img_paths = [
    #         'tests/data/det_toy_dataset/imgs/test/img_1.jpg',
    #         'tests/data/det_toy_dataset/imgs/test/img_2.jpg'
    #     ]
    #     res_path = self.inferencer(img_paths, return_vis=True)
    #     # list of ndarray
    #     imgs = [mmcv.imread(p) for p in img_paths]
    #     res_ndarray = self.inferencer(imgs, return_vis=True)
    #     self.assert_predictions_equal(res_path['predictions'],
    #                                   res_ndarray['predictions'])
    #     self.assertIn('visualization', res_path)
    #     self.assertIn('visualization', res_ndarray)

    #     # img dir, test different batch sizes
    #     img_dir = 'tests/data/det_toy_dataset/imgs/test/'
    #     res_bs1 = self.inferencer(img_dir, batch_size=1, return_vis=True)
    #     res_bs3 = self.inferencer(img_dir, batch_size=3, return_vis=True)
    #     self.assert_predictions_equal(res_bs1['predictions'],
    #                                   res_bs3['predictions'])
    #     self.assertTrue(
    #         np.array_equal(res_bs1['visualization'], res_bs3['visualization']))

    # def test_visualize(self):
    #     img_paths = [
    #         'tests/data/det_toy_dataset/imgs/test/img_1.jpg',
    #         'tests/data/det_toy_dataset/imgs/test/img_2.jpg'
    #     ]

    #     # img_out_dir
    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         self.inferencer(img_paths, img_out_dir=tmp_dir)
    #         for img_dir in ['img_1.jpg', 'img_2.jpg']:
    #             self.assertTrue(osp.exists(osp.join(tmp_dir, img_dir)))

    # def test_postprocess(self):
    #     # return_datasample
    #     img_path = 'tests/data/det_toy_dataset/imgs/test/img_1.jpg'
    #     res = self.inferencer(img_path, return_datasamples=True)
    #     self.assertTrue(is_type_list(res['predictions'], TextDetDataSample))

    #     # pred_out_file
    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         pred_out_file = osp.join(tmp_dir, 'tmp.pkl')
    #         res = self.inferencer(
    #             img_path, print_result=True, pred_out_file=pred_out_file)
    #         dumped_res = mmengine.load(pred_out_file)
    #         self.assert_predictions_equal(res['predictions'],
    #                                       dumped_res['predictions'])
