# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import random
import tempfile
from unittest import TestCase, mock

import mmcv
import mmengine
import numpy as np
import torch

from mmocr.apis.inferencers import TextRecInferencer
from mmocr.utils.check_argument import is_type_list
from mmocr.utils.typing_utils import TextRecogDataSample


class TestTextRecinferencer(TestCase):

    @mock.patch('mmengine.infer.infer._load_checkpoint')
    def setUp(self, mock_load):
        mock_load.side_effect = lambda *x, **y: None
        # init from alias
        self.inferencer = TextRecInferencer('CRNN')
        seed = 1
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @mock.patch('mmengine.infer.infer._load_checkpoint')
    def test_init(self, mock_load):
        mock_load.side_effect = lambda *x, **y: None
        # init from metafile
        TextRecInferencer('crnn_mini-vgg_5e_mj')
        # init from cfg
        TextRecInferencer(
            'configs/textrecog/crnn/crnn_mini-vgg_5e_mj.py',
            'https://download.openmmlab.com/mmocr/textrecog/crnn/'
            'crnn_mini-vgg_5e_mj/'
            'crnn_mini-vgg_5e_mj_20220826_224120-8afbedbb.pth')

    def assert_predictions_equal(self, preds1, preds2):
        for pred1, pred2 in zip(preds1, preds2):
            self.assert_prediction_equal(pred1, pred2)

    def assert_prediction_equal(self, pred1, pred2):
        self.assertEqual(pred1['text'], pred2['text'])
        self.assertTrue(np.allclose(pred1['scores'], pred2['scores'], 0.1))

    def test_call(self):
        # single img
        img_path = 'tests/data/rec_toy_dataset/imgs/1036169.jpg'
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
            'tests/data/rec_toy_dataset/imgs/1036169.jpg',
            'tests/data/rec_toy_dataset/imgs/1058891.jpg'
        ]
        res_path = self.inferencer(img_paths, return_vis=True)
        # list of ndarray
        imgs = [mmcv.imread(p) for p in img_paths]
        res_ndarray = self.inferencer(imgs, return_vis=True)
        self.assert_predictions_equal(res_path['predictions'],
                                      res_ndarray['predictions'])
        for i in range(len(img_paths)):
            self.assertTrue(
                np.allclose(res_path['visualization'][i],
                            res_ndarray['visualization'][i]))

        # img dir, test different batch sizes
        img_dir = 'tests/data/rec_toy_dataset/imgs'
        res_bs3 = self.inferencer(img_dir, batch_size=3, return_vis=True)
        self.assertIn('visualization', res_bs3)
        self.assertIn('predictions', res_bs3)

    def test_visualize(self):
        img_paths = [
            'tests/data/rec_toy_dataset/imgs/1036169.jpg',
            'tests/data/rec_toy_dataset/imgs/1058891.jpg'
        ]

        # img_out_dir
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.inferencer(img_paths, out_dir=tmp_dir, save_vis=True)
            for img_dir in ['1036169.jpg', '1058891.jpg']:
                self.assertTrue(osp.exists(osp.join(tmp_dir, 'vis', img_dir)))

    def test_postprocess(self):
        # return_datasample
        img_path = 'tests/data/rec_toy_dataset/imgs/1036169.jpg'
        res = self.inferencer(img_path, return_datasamples=True)
        self.assertTrue(is_type_list(res['predictions'], TextRecogDataSample))

        # pred_out_file
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = self.inferencer(
                img_path, print_result=True, out_dir=tmp_dir, save_pred=True)
            dumped_res = mmengine.load(
                osp.join(tmp_dir, 'preds', '1036169.json'))
            self.assert_prediction_equal(res['predictions'][0], dumped_res)
