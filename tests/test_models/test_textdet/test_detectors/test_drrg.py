# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import mock

import numpy as np
import torch
from base import BaseTestUtils

from mmocr.registry import MODELS
from mmocr.utils import register_all_modules


class TestDRRG(BaseTestUtils, unittest.TestCase):

    def setUp(self):
        cfg_path = 'textdet/drrg/drrg_r50_fpn_unet_1200e_ctw1500.py'
        self.model_cfg = self._get_detector_cfg(cfg_path)
        register_all_modules()
        self.model = MODELS.build(self.model_cfg)
        self.inputs = self._create_dummy_inputs(input_shape=(1, 3, 224, 224))

    def _get_comp_attribs(self):
        num_rois = 32
        x = np.random.randint(4, 224, (num_rois, 1))
        y = np.random.randint(4, 224, (num_rois, 1))
        h = 4 * np.ones((num_rois, 1))
        w = 4 * np.ones((num_rois, 1))
        angle = (np.random.random_sample((num_rois, 1)) * 2 - 1) * np.pi / 2
        cos, sin = np.cos(angle), np.sin(angle)
        comp_labels = np.random.randint(1, 3, (num_rois, 1))
        num_rois = num_rois * np.ones((num_rois, 1))
        comp_attribs = np.hstack([num_rois, x, y, h, w, cos, sin, comp_labels])
        gt_comp_attribs = np.expand_dims(
            comp_attribs.astype(np.float32), axis=0)
        return gt_comp_attribs

    def _get_drrg_inputs(self):
        imgs = self.inputs['imgs']
        data_samples = self.inputs['data_samples']
        gt_text_mask = self.inputs['gt_text_mask']
        gt_center_region_mask = self.inputs['gt_center_region_mask']
        gt_mask = self.inputs['gt_mask']
        gt_top_height_map = self.inputs['gt_radius_map']
        gt_bot_height_map = gt_top_height_map.copy()
        gt_sin_map = self.inputs['gt_sin_map']
        gt_cos_map = self.inputs['gt_cos_map']
        gt_comp_attribs = self._get_comp_attribs()
        return imgs, data_samples, (gt_text_mask, gt_center_region_mask,
                                    gt_mask, gt_top_height_map,
                                    gt_bot_height_map, gt_sin_map, gt_cos_map,
                                    gt_comp_attribs)

    @mock.patch('mmocr.models.textdet.losses.drrg_loss.DRRGLoss.get_targets')
    def test_loss(self, mock_get_targets):
        imgs, data_samples, targets = self._get_drrg_inputs()
        mock_get_targets.return_value = targets
        losses = self.model(imgs, data_samples, mode='loss')
        self.assertIsInstance(losses, dict)

    @mock.patch('mmocr.models.textdet.detectors.drrg.DRRG.extract_feat')
    def test_predict(self, mock_extract_feat):
        model_cfg = self.model_cfg.copy()
        model_cfg['det_head']['in_channels'] = 6
        model_cfg['det_head']['text_region_thr'] = 0.8
        model_cfg['det_head']['center_region_thr'] = 0.8
        model = MODELS.build(model_cfg)
        imgs, data_samples, _ = self._get_drrg_inputs()

        maps = torch.zeros((1, 6, 224, 224), dtype=torch.float)
        maps[:, 0:2, :, :] = -10.
        maps[:, 0, 60:100, 50:170] = 10.
        maps[:, 1, 75:85, 60:160] = 10.
        maps[:, 2, 75:85, 60:160] = 0.
        maps[:, 3, 75:85, 60:160] = 1.
        maps[:, 4, 75:85, 60:160] = 10.
        maps[:, 5, 75:85, 60:160] = 10.
        mock_extract_feat.return_value = maps
        with torch.no_grad():
            full_pass_weight = torch.zeros((6, 6, 1, 1))
            for i in range(6):
                full_pass_weight[i, i, 0, 0] = 1
            model.det_head.out_conv.weight.data = full_pass_weight
            model.det_head.out_conv.bias.data.fill_(0.)
            results = model(imgs, data_samples, mode='predict')
        self.assertIn('polygons', results[0].pred_instances)
        self.assertIn('scores', results[0].pred_instances)
        self.assertTrue(
            isinstance(results[0].pred_instances['scores'], torch.FloatTensor))
