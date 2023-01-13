# Copyright (c) OpenMMLab. All rights reserved.

import math
from unittest import TestCase
from unittest.mock import patch

import torch
from mmengine.logging import MMLogger

from mmocr.datasets import RepeatAugSampler

file = 'mmocr.datasets.samplers.repeat_aug.'


class MockDist:

    def __init__(self, dist_info=(0, 1), seed=7):
        self.dist_info = dist_info
        self.seed = seed

    def get_dist_info(self):
        return self.dist_info

    def sync_random_seed(self):
        return self.seed

    def is_main_process(self):
        return self.dist_info[0] == 0


class TestRepeatAugSampler(TestCase):

    def setUp(self):
        self.data_length = 100
        self.dataset = list(range(self.data_length))

    @patch(file + 'get_dist_info', return_value=(0, 1))
    def test_non_dist(self, mock):
        sampler = RepeatAugSampler(self.dataset, num_repeats=3, shuffle=False)
        self.assertEqual(sampler.world_size, 1)
        self.assertEqual(sampler.rank, 0)
        self.assertEqual(sampler.total_size, self.data_length * 3)
        self.assertEqual(sampler.num_samples, self.data_length * 3)
        self.assertEqual(sampler.num_selected_samples, self.data_length)
        self.assertEqual(len(sampler), sampler.num_selected_samples)
        indices = [x for x in range(self.data_length) for _ in range(3)]
        self.assertEqual(list(sampler), indices[:self.data_length])

        logger = MMLogger.get_current_instance()
        with self.assertLogs(logger, 'WARN') as log:
            sampler = RepeatAugSampler(self.dataset, shuffle=False)
        self.assertIn('always picks a fixed part', log.output[0])

    @patch(file + 'get_dist_info', return_value=(2, 3))
    @patch(file + 'is_main_process', return_value=False)
    def test_dist(self, mock1, mock2):
        sampler = RepeatAugSampler(self.dataset, num_repeats=3, shuffle=False)
        self.assertEqual(sampler.world_size, 3)
        self.assertEqual(sampler.rank, 2)
        self.assertEqual(sampler.num_samples, self.data_length)
        self.assertEqual(sampler.total_size, self.data_length * 3)
        self.assertEqual(sampler.num_selected_samples,
                         math.ceil(self.data_length / 3))
        self.assertEqual(len(sampler), sampler.num_selected_samples)
        indices = [x for x in range(self.data_length) for _ in range(3)]
        self.assertEqual(
            list(sampler), indices[2::3][:sampler.num_selected_samples])

        logger = MMLogger.get_current_instance()
        with patch.object(logger, 'warning') as mock_log:
            sampler = RepeatAugSampler(self.dataset, shuffle=False)
            mock_log.assert_not_called()

    @patch(file + 'get_dist_info', return_value=(0, 1))
    @patch(file + 'sync_random_seed', return_value=7)
    def test_shuffle(self, mock1, mock2):
        # test seed=None
        sampler = RepeatAugSampler(self.dataset, seed=None)
        self.assertEqual(sampler.seed, 7)

        # test random seed
        sampler = RepeatAugSampler(self.dataset, shuffle=True, seed=0)
        sampler.set_epoch(10)
        g = torch.Generator()
        g.manual_seed(10)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        indices = [x for x in indices
                   for _ in range(3)][:sampler.num_selected_samples]
        self.assertEqual(list(sampler), indices)

        sampler = RepeatAugSampler(self.dataset, shuffle=True, seed=42)
        sampler.set_epoch(10)
        g = torch.Generator()
        g.manual_seed(42 + 10)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        indices = [x for x in indices
                   for _ in range(3)][:sampler.num_selected_samples]
        self.assertEqual(list(sampler), indices)
