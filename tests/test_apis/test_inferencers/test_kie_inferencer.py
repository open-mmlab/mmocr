# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import random
import tempfile
from copy import deepcopy
from unittest import TestCase, mock

import mmcv
import mmengine
import numpy as np
import torch

from mmocr.apis.inferencers import KIEInferencer
from mmocr.utils.check_argument import is_type_list
from mmocr.utils.polygon_utils import poly2bbox
from mmocr.utils.typing_utils import KIEDataSample


class TestKIEInferencer(TestCase):

    @mock.patch('mmengine.infer.infer._load_checkpoint')
    def setUp(self, mock_load):
        mock_load.side_effect = lambda *x, **y: None
        seed = 1
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # init from alias
        self.inferencer = KIEInferencer('SDMGR')
        self.inferencer_novisual = KIEInferencer(
            'sdmgr_novisual_60e_wildreceipt')
        with open('tests/data/kie_toy_dataset/wildreceipt/data.txt', 'r') as f:
            annos = [json.loads(anno) for anno in f.readlines()]

        self.data_novisual = []
        self.data_img_str = []
        self.data_img_ndarray = []
        self.data_img_woshape = []

        for anno in annos:
            datum_novisual = dict(img_shape=(anno['height'], anno['width']))
            datum_novisual['instances'] = []
            for ann in anno['annotations']:
                instance = {}
                instance['bbox'] = poly2bbox(
                    np.array(ann['box'], dtype=np.float32))
                instance['text'] = ann['text']
                datum_novisual['instances'].append(instance)
            self.data_novisual.append(datum_novisual)

            datum_img_str = deepcopy(datum_novisual)
            datum_img_str['img'] = anno['file_name']
            self.data_img_str.append(datum_img_str)

            datum_img_ndarray = deepcopy(datum_novisual)
            datum_img_ndarray['img'] = mmcv.imread(anno['file_name'])
            self.data_img_ndarray.append(datum_img_ndarray)

            datum_img_woshape = deepcopy(datum_img_str)
            del datum_img_woshape['img_shape']
            self.data_img_woshape.append(datum_img_woshape)

    @mock.patch('mmengine.infer.infer._load_checkpoint')
    def test_init(self, mock_load):
        mock_load.side_effect = lambda *x, **y: None
        # init from metafile
        KIEInferencer('sdmgr_unet16_60e_wildreceipt')
        # init from cfg
        KIEInferencer(
            'configs/kie/sdmgr/sdmgr_unet16_60e_wildreceipt.py',
            'https://download.openmmlab.com/mmocr/kie/sdmgr/'
            'sdmgr_unet16_60e_wildreceipt/'
            'sdmgr_unet16_60e_wildreceipt_20220825_151648-22419f37.pth')

    def assert_predictions_equal(self, preds1, preds2):
        for pred1, pred2 in zip(preds1, preds2):
            self.assert_prediction_equal(pred1, pred2)

    def assert_prediction_equal(self, pred1, pred2):
        self.assertTrue(np.allclose(pred1['labels'], pred2['labels'], 0.1))
        self.assertTrue(
            np.allclose(pred1['edge_scores'], pred2['edge_scores'], 0.1))
        self.assertTrue(
            np.allclose(pred1['edge_labels'], pred2['edge_labels'], 0.1))
        self.assertTrue(np.allclose(pred1['scores'], pred2['scores'], 0.1))

    def test_call(self):
        # no visual, single input
        res_novis_1 = self.inferencer_novisual(
            self.data_novisual[0], return_vis=True)
        res_novis_2 = self.inferencer_novisual(
            self.data_img_woshape[0], return_vis=True)
        self.assert_predictions_equal(res_novis_1['predictions'],
                                      res_novis_2['predictions'])
        self.assertIn('visualization', res_novis_1)
        self.assertIn('visualization', res_novis_2)

        # no visual, multiple inputs
        res_novis_1 = self.inferencer_novisual(
            self.data_novisual, return_vis=True)
        res_novis_2 = self.inferencer_novisual(
            self.data_img_woshape, return_vis=True)
        self.assert_predictions_equal(res_novis_1['predictions'],
                                      res_novis_2['predictions'])
        self.assertIn('visualization', res_novis_1)
        self.assertIn('visualization', res_novis_2)

        # visual, single input
        res_ndarray = self.inferencer(
            self.data_img_ndarray[0], return_vis=True)
        # path
        res_path = self.inferencer(self.data_img_str[0], return_vis=True)
        self.assert_predictions_equal(res_path['predictions'],
                                      res_ndarray['predictions'])
        self.assertIn('visualization', res_path)
        self.assertIn('visualization', res_ndarray)
        self.assertTrue(
            np.allclose(res_ndarray['visualization'],
                        res_path['visualization']))

        # visual, multiple inputs & different bs
        res_ndarray = self.inferencer(self.data_img_ndarray, return_vis=True)
        # path
        res_path = self.inferencer(self.data_img_str, return_vis=True)
        self.assert_predictions_equal(res_path['predictions'],
                                      res_ndarray['predictions'])
        for vis1, vis2 in zip(res_ndarray['visualization'],
                              res_path['visualization']):
            self.assertTrue(np.allclose(vis1, vis2))

    def test_visualize(self):

        # img_out_dir
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.inferencer(self.data_img_str, out_dir=tmp_dir, save_vis=True)
            for img_dir in ['1.jpg', '2.jpg']:
                self.assertTrue(osp.exists(osp.join(tmp_dir, 'vis', img_dir)))

    def test_postprocess(self):
        # return_datasample
        res = self.inferencer(self.data_img_ndarray, return_datasamples=True)
        self.assertTrue(is_type_list(res['predictions'], KIEDataSample))

        # pred_out_file
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = self.inferencer(
                self.data_img_ndarray,
                print_result=True,
                out_dir=tmp_dir,
                save_pred=True)
            file_names = [
                f'{self.inferencer.num_unnamed_imgs - i}.json'
                for i in range(len(self.data_img_ndarray), 0, -1)
            ]
            for pred, file_name in zip(res['predictions'], file_names):
                dumped_res = mmengine.load(
                    osp.join(tmp_dir, 'preds', file_name))
                self.assert_prediction_equal(dumped_res, pred)

    @mock.patch('mmocr.apis.inferencers.kie_inferencer._load_checkpoint')
    def test_load_metainfo_to_visualizer(self, mock_load):
        mock_load.side_effect = lambda *x, **y: {'meta': 'test'}
        with self.assertRaises(ValueError):
            self.inferencer._load_metainfo_to_visualizer('test', {})

        mock_load.side_effect = lambda *x, **y: {
            'meta': {
                'dataset_meta': 'test'
            }
        }
        self.inferencer._load_metainfo_to_visualizer('test', {})

        with self.assertRaises(ValueError):
            self.inferencer._load_metainfo_to_visualizer(None, {})
