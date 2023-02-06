# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import random
import tempfile
from unittest import TestCase, mock

import mmcv
import mmengine
import numpy as np
import torch
from mmengine.structures import InstanceData

from mmocr.apis.inferencers import TextDetInferencer
from mmocr.utils.check_argument import is_type_list
from mmocr.utils.typing_utils import TextDetDataSample


class TestTextDetinferencer(TestCase):

    @mock.patch('mmengine.infer.infer._load_checkpoint')
    def setUp(self, mock_load):
        mock_load.side_effect = lambda *x, **y: None
        self.inferencer = TextDetInferencer('DB_r18')
        seed = 1
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @mock.patch('mmengine.infer.infer._load_checkpoint')
    def test_init(self, mock_load):
        mock_load.side_effect = lambda *x, **y: None
        # init from metafile
        TextDetInferencer('dbnet_resnet18_fpnc_1200e_icdar2015')
        # init from cfg
        TextDetInferencer(
            'configs/textdet/dbnet/'
            'dbnet_resnet18_fpnc_1200e_icdar2015.py',
            'https://download.openmmlab.com/mmocr/textdet/dbnet/'
            'dbnet_resnet18_fpnc_1200e_icdar2015/'
            'dbnet_resnet18_fpnc_1200e_icdar2015_20220825_221614-7c0e94f2.pth')

    def assert_predictions_equal(self, preds1, preds2):
        for pred1, pred2 in zip(preds1, preds2):
            self.assertTrue(
                np.allclose(pred1['polygons'], pred2['polygons'], 0.1))
            self.assertTrue(np.allclose(pred1['scores'], pred2['scores'], 0.1))

    def test_call(self):
        # single img
        img_path = 'tests/data/det_toy_dataset/imgs/test/img_1.jpg'
        res_path = self.inferencer(img_path, return_vis=True)
        # ndarray
        img = mmcv.imread(img_path)
        res_ndarray = self.inferencer(img, return_vis=True)
        self.assert_predictions_equal(res_path['predictions'],
                                      res_ndarray['predictions'])
        self.assertTrue(
            np.allclose(res_path['visualization'],
                        res_ndarray['visualization']))

        # multiple images
        img_paths = [
            'tests/data/det_toy_dataset/imgs/test/img_1.jpg',
            'tests/data/det_toy_dataset/imgs/test/img_2.jpg'
        ]
        res_path = self.inferencer(img_paths, return_vis=True)
        # list of ndarray
        imgs = [mmcv.imread(p) for p in img_paths]
        res_ndarray = self.inferencer(imgs, return_vis=True)
        self.assert_predictions_equal(res_path['predictions'],
                                      res_ndarray['predictions'])
        self.assertTrue(
            np.allclose(res_path['visualization'],
                        res_ndarray['visualization']))

        # img dir, test different batch sizes
        img_dir = 'tests/data/det_toy_dataset/imgs/test/'
        res_bs1 = self.inferencer(img_dir, batch_size=1, return_vis=True)
        res_bs3 = self.inferencer(img_dir, batch_size=3, return_vis=True)
        self.assert_predictions_equal(res_bs1['predictions'],
                                      res_bs3['predictions'])
        self.assertTrue(
            np.array_equal(res_bs1['visualization'], res_bs3['visualization']))

    def test_visualize(self):
        img_paths = [
            'tests/data/det_toy_dataset/imgs/test/img_1.jpg',
            'tests/data/det_toy_dataset/imgs/test/img_2.jpg'
        ]

        # img_out_dir
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.inferencer(img_paths, img_out_dir=tmp_dir)
            for img_dir in ['img_1.jpg', 'img_2.jpg']:
                self.assertTrue(osp.exists(osp.join(tmp_dir, img_dir)))

    def test_postprocess(self):
        # return_datasample
        img_path = 'tests/data/det_toy_dataset/imgs/test/img_1.jpg'
        res = self.inferencer(img_path, return_datasamples=True)
        self.assertTrue(is_type_list(res['predictions'], TextDetDataSample))

        # pred_out_file
        with tempfile.TemporaryDirectory() as tmp_dir:
            pred_out_file = osp.join(tmp_dir, 'tmp.pkl')
            res = self.inferencer(
                img_path, print_result=True, pred_out_file=pred_out_file)
            dumped_res = mmengine.load(pred_out_file)
            self.assert_predictions_equal(res['predictions'],
                                          dumped_res['predictions'])

    def test_pred2dict(self):
        data_sample = TextDetDataSample()
        data_sample.pred_instances = InstanceData()

        data_sample.pred_instances.scores = np.array([0.9])
        data_sample.pred_instances.polygons = [
            np.array([0, 0, 0, 1, 1, 1, 1, 0])
        ]
        res = self.inferencer.pred2dict(data_sample)
        self.assertListAlmostEqual(res['polygons'], [[0, 0, 0, 1, 1, 1, 1, 0]])
        self.assertListAlmostEqual(res['scores'], [0.9])

        data_sample.pred_instances.bboxes = np.array([[0, 0, 1, 1]])
        data_sample.pred_instances.scores = torch.FloatTensor([0.9])
        res = self.inferencer.pred2dict(data_sample)
        self.assertListAlmostEqual(res['polygons'], [[0, 0, 0, 1, 1, 1, 1, 0]])
        self.assertListAlmostEqual(res['bboxes'], [[0, 0, 1, 1]])
        self.assertListAlmostEqual(res['scores'], [0.9])

    def assertListAlmostEqual(self, list1, list2, places=7):
        for i in range(len(list1)):
            if isinstance(list1[i], list):
                self.assertListAlmostEqual(list1[i], list2[i], places=places)
            else:
                self.assertAlmostEqual(list1[i], list2[i], places=places)
