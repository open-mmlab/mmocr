# Copyright (c) OpenMMLab. All rights reserved.
import copy
import shutil
import tempfile
from unittest import TestCase

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.evaluator import BaseMetric
from mmengine.hooks import Hook
from mmengine.model import BaseModel
from mmengine.registry import DATASETS, HOOKS, METRICS, MODELS
from mmengine.runner import Runner
from torch.utils.data import Dataset

from mmocr.engine.runner import MultiTestLoop, MultiValLoop


@MODELS.register_module()
class ToyModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, batch_inputs, labels, mode='tensor'):
        labels = torch.stack(labels)
        outputs = self.linear1(batch_inputs)
        outputs = self.linear2(outputs)

        if mode == 'tensor':
            return outputs
        elif mode == 'loss':
            loss = (labels - outputs).sum()
            outputs = dict(loss=loss)
            return outputs
        elif mode == 'predict':
            return outputs


@DATASETS.register_module()
class ToyDataset(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    @property
    def metainfo(self):
        return self.METAINFO

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_sample=self.label[index])


@METRICS.register_module()
class ToyMetric3(BaseMetric):

    def __init__(self, collect_device='cpu', prefix=''):
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_samples, predictions):
        result = {'acc': 1}
        self.results.append(result)

    def compute_metrics(self, results):
        return dict(acc=1)


class TestRunner(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        epoch_based_cfg = dict(
            default_scope='mmocr',
            model=dict(type='ToyModel'),
            work_dir=self.temp_dir,
            train_dataloader=dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=3,
                num_workers=0),
            val_dataloader=dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=3,
                num_workers=0),
            test_dataloader=dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=3,
                num_workers=0),
            auto_scale_lr=dict(base_batch_size=16, enable=False),
            optim_wrapper=dict(
                type='OptimWrapper', optimizer=dict(type='SGD', lr=0.01)),
            param_scheduler=dict(type='MultiStepLR', milestones=[1, 2]),
            val_evaluator=dict(type='ToyMetric1'),
            test_evaluator=dict(type='ToyMetric1'),
            train_cfg=dict(
                by_epoch=True, max_epochs=3, val_interval=1, val_begin=1),
            val_cfg=dict(),
            test_cfg=dict(),
            custom_hooks=[],
            default_hooks=dict(
                runtime_info=dict(type='RuntimeInfoHook'),
                timer=dict(type='IterTimerHook'),
                logger=dict(type='LoggerHook'),
                param_scheduler=dict(type='ParamSchedulerHook'),
                checkpoint=dict(
                    type='CheckpointHook', interval=1, by_epoch=True),
                sampler_seed=dict(type='DistSamplerSeedHook')),
            launcher='none',
            env_cfg=dict(dist_cfg=dict(backend='nccl')),
        )
        self.epoch_based_cfg = Config(epoch_based_cfg)
        self.iter_based_cfg = copy.deepcopy(self.epoch_based_cfg)
        self.iter_based_cfg.train_dataloader = dict(
            dataset=dict(type='ToyDataset'),
            sampler=dict(type='InfiniteSampler', shuffle=True),
            batch_size=3,
            num_workers=0)
        self.iter_based_cfg.train_cfg = dict(by_epoch=False, max_iters=12)
        self.iter_based_cfg.default_hooks = dict(
            runtime_info=dict(type='RuntimeInfoHook'),
            timer=dict(type='IterTimerHook'),
            logger=dict(type='LoggerHook'),
            param_scheduler=dict(type='ParamSchedulerHook'),
            checkpoint=dict(type='CheckpointHook', interval=1, by_epoch=False),
            sampler_seed=dict(type='DistSamplerSeedHook'))

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_multi_val_loop(self):

        before_val_iter_results = []
        after_val_iter_results = []
        multi_metrics = dict()

        @HOOKS.register_module()
        class Fake_1(Hook):
            """test custom train loop."""

            def before_val_iter(self, runner, batch_idx, data_batch=None):
                before_val_iter_results.append('before')

            def after_val_iter(self,
                               runner,
                               batch_idx,
                               data_batch=None,
                               outputs=None):
                after_val_iter_results.append('after')

            def after_val_epoch(self, runner, metrics=None) -> None:
                multi_metrics.update(metrics)

        self.iter_based_cfg.val_cfg = dict(type='MultiValLoop')
        self.iter_based_cfg.val_dataloader = [
            dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=3,
                num_workers=0),
            dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=3,
                num_workers=0)
        ]
        self.iter_based_cfg.val_evaluator = [
            dict(type='ToyMetric3', prefix='tmp1'),
            dict(type='ToyMetric3', prefix='tmp2')
        ]
        self.iter_based_cfg.custom_hooks = [dict(type='Fake_1', priority=50)]
        self.iter_based_cfg.experiment_name = 'test_multi_val_loop'
        runner = Runner.from_cfg(self.iter_based_cfg)
        runner.val()

        self.assertIsInstance(runner.val_loop, MultiValLoop)

        # test custom hook triggered as expected
        self.assertEqual(len(before_val_iter_results), 8)
        self.assertEqual(len(after_val_iter_results), 8)
        for before, after in zip(before_val_iter_results,
                                 after_val_iter_results):
            self.assertEqual(before, 'before')
            self.assertEqual(after, 'after')
        self.assertDictEqual(multi_metrics, {'tmp1/acc': 1, 'tmp2/acc': 1})

        # test_same prefix
        self.iter_based_cfg.val_evaluator = [
            dict(type='ToyMetric3', prefix='tmp1'),
            dict(type='ToyMetric3', prefix='tmp1')
        ]
        self.iter_based_cfg.experiment_name = 'test_multi_val_loop_same_prefix'
        runner = Runner.from_cfg(self.iter_based_cfg)
        with self.assertRaisesRegex(ValueError,
                                    ('Please set different'
                                     ' prefix for different datasets'
                                     ' in `val_evaluator`')):
            runner.val()

    def test_multi_test_loop(self):

        before_test_iter_results = []
        after_test_iter_results = []
        multi_metrics = dict()

        @HOOKS.register_module()
        class Fake_2(Hook):
            """test custom train loop."""

            def before_test_iter(self, runner, batch_idx, data_batch=None):
                before_test_iter_results.append('before')

            def after_test_iter(self,
                                runner,
                                batch_idx,
                                data_batch=None,
                                outputs=None):
                after_test_iter_results.append('after')

            def after_test_epoch(self, runner, metrics=None) -> None:
                multi_metrics.update(metrics)

        self.iter_based_cfg.test_cfg = dict(type='MultiTestLoop')
        self.iter_based_cfg.test_dataloader = [
            dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=3,
                num_workers=0),
            dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=3,
                num_workers=0)
        ]
        self.iter_based_cfg.test_evaluator = [
            dict(type='ToyMetric3', prefix='tmp1'),
            dict(type='ToyMetric3', prefix='tmp2')
        ]
        self.iter_based_cfg.custom_hooks = [dict(type='Fake_2', priority=50)]
        self.iter_based_cfg.experiment_name = 'multi_test_loop'
        runner = Runner.from_cfg(self.iter_based_cfg)
        runner.test()

        self.assertIsInstance(runner.test_loop, MultiTestLoop)

        # test custom hook triggered as expected
        self.assertEqual(len(before_test_iter_results), 8)
        self.assertEqual(len(after_test_iter_results), 8)
        for before, after in zip(before_test_iter_results,
                                 after_test_iter_results):
            self.assertEqual(before, 'before')
            self.assertEqual(after, 'after')
        self.assertDictEqual(multi_metrics, {'tmp1/acc': 1, 'tmp2/acc': 1})

        # test_same prefix
        self.iter_based_cfg.test_evaluator = [
            dict(type='ToyMetric3', prefix='tmp1'),
            dict(type='ToyMetric3', prefix='tmp1')
        ]
        self.iter_based_cfg.experiment_name = 'multi_test_loop_same_prefix'
        runner = Runner.from_cfg(self.iter_based_cfg)
        with self.assertRaisesRegex(ValueError,
                                    ('Please set different'
                                     ' prefix for different datasets'
                                     ' in `test_evaluator`')):
            runner.test()
