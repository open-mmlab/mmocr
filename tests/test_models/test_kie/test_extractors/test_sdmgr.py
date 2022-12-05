# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest
from os.path import dirname, exists, join

import torch
from mmengine.config import Config, ConfigDict
from mmengine.structures import InstanceData

from mmocr.registry import MODELS
from mmocr.structures import KIEDataSample


class TestSDMGR(unittest.TestCase):

    def _get_config_directory(self):
        """Find the predefined detector config directory."""
        try:
            # Assume we are running in the source mmocr repo
            repo_dpath = dirname(dirname(dirname(dirname(dirname(__file__)))))
        except NameError:
            # For IPython development when this __file__ is not defined
            import mmocr
            repo_dpath = dirname(
                dirname(dirname(dirname(dirname(mmocr.__file__)))))
        config_dpath = join(repo_dpath, 'configs')
        if not exists(config_dpath):
            raise Exception('Cannot find config path')
        return config_dpath

    def _get_config_module(self, fname: str) -> 'ConfigDict':
        """Load a configuration as a python module."""
        config_dpath = self._get_config_directory()
        config_fpath = join(config_dpath, fname)
        config_mod = Config.fromfile(config_fpath)
        return config_mod

    def _get_cfg(self, fname: str) -> 'ConfigDict':
        """Grab configs necessary to create a detector.

        These are deep copied to allow for safe modification of parameters
        without influencing other tests.
        """
        config = self._get_config_module(fname)
        model = copy.deepcopy(config.model)
        model.dictionary.dict_file = 'dicts/lower_english_digits.txt'
        return model

    def forward_wrapper(self, model, data, mode):
        out = model.data_preprocessor(data, False)
        inputs, data_samples = out['inputs'], out['data_samples']
        return model.forward(inputs, data_samples, mode)

    def setUp(self):

        cfg_path = 'kie/sdmgr/sdmgr_unet16_60e_wildreceipt.py'
        self.visual_model_cfg = self._get_cfg(cfg_path)
        self.visual_model = MODELS.build(self.visual_model_cfg)

        cfg_path = 'kie/sdmgr/sdmgr_novisual_60e_wildreceipt.py'
        self.novisual_model_cfg = self._get_cfg(cfg_path)
        self.novisual_model = MODELS.build(self.novisual_model_cfg)

        data_sample = KIEDataSample()
        data_sample.gt_instances = InstanceData(
            bboxes=torch.FloatTensor([[0, 0, 1, 1], [1, 1, 2, 2]]),
            labels=torch.LongTensor([0, 1]),
            edge_labels=torch.LongTensor([[0, 1], [1, 0]]),
            texts=['text1', 'text2'],
            relations=torch.rand((2, 2, 5)))
        self.visual_data = dict(
            inputs=[torch.rand((3, 10, 10))], data_samples=[data_sample])
        self.novisual_data = dict(
            inputs=[torch.Tensor([]).reshape((0, 0, 0))],
            data_samples=[data_sample])

    def test_forward_loss(self):
        result = self.forward_wrapper(
            self.visual_model, self.visual_data, mode='loss')
        self.assertIsInstance(result, dict)

        result = self.forward_wrapper(
            self.novisual_model, self.visual_data, mode='loss')
        self.assertIsInstance(result, dict)

    def test_forward_predict(self):
        result = self.forward_wrapper(
            self.visual_model, self.visual_data, mode='predict')[0]
        self.assertIsInstance(result, KIEDataSample)
        self.assertEqual(result.pred_instances.labels.shape, torch.Size([2]))
        self.assertEqual(result.pred_instances.edge_labels.shape,
                         torch.Size([2, 2]))

        result = self.forward_wrapper(
            self.novisual_model, self.novisual_data, mode='predict')[0]
        self.assertIsInstance(result, KIEDataSample)
        self.assertEqual(result.pred_instances.labels.shape, torch.Size([2]))
        self.assertEqual(result.pred_instances.edge_labels.shape,
                         torch.Size([2, 2]))

    def test_forward_tensor(self):
        result = self.forward_wrapper(
            self.visual_model, self.visual_data, mode='tensor')
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], torch.Tensor)
        self.assertIsInstance(result[1], torch.Tensor)

        result = self.forward_wrapper(
            self.novisual_model, self.novisual_data, mode='tensor')
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], torch.Tensor)
        self.assertIsInstance(result[1], torch.Tensor)

    def test_forward_invalid(self):
        with self.assertRaises(RuntimeError):
            self.forward_wrapper(
                self.visual_model, self.visual_data, mode='invalid')
