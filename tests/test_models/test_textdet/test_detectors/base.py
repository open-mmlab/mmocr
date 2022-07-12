# Copyright (c) OpenMMLab. All rights reserved.
import copy
from os.path import dirname, exists, join

import numpy as np
import torch
from mmengine import Config, ConfigDict, InstanceData

from mmocr.data import TextDetDataSample


class BaseTestUtils:

    def _create_dummy_inputs(self,
                             input_shape=(1, 3, 300, 300),
                             num_items=None,
                             num_classes=1):
        """Create a superset of inputs needed to run test or train batches."""
        (N, C, H, W) = input_shape

        rng = np.random.RandomState(0)

        imgs = rng.rand(*input_shape)

        metainfo = dict(
            img_shape=(H, W, C),
            ori_shape=(H, W, C),
            pad_shape=(H, W, C),
            filename='test.jpg',
            scale_factor=(1, 1),
            flip=False)

        gt_masks = []
        gt_kernels = []
        gt_effective_mask = []

        data_samples = []

        for batch_idx in range(N):
            if num_items is None:
                num_boxes = rng.randint(1, 10)
            else:
                num_boxes = num_items[batch_idx]

            data_sample = TextDetDataSample(
                metainfo=metainfo, gt_instances=InstanceData())

            cx, cy, bw, bh = rng.rand(num_boxes, 4).T

            tl_x = ((cx * W) - (W * bw / 2)).clip(0, W)
            tl_y = ((cy * H) - (H * bh / 2)).clip(0, H)
            br_x = ((cx * W) + (W * bw / 2)).clip(0, W)
            br_y = ((cy * H) + (H * bh / 2)).clip(0, H)

            boxes = np.vstack([tl_x, tl_y, br_x, br_y]).T
            class_idxs = [0] * num_boxes

            data_sample.gt_instances.bboxes = torch.FloatTensor(boxes)
            data_sample.gt_instances.labels = torch.LongTensor(class_idxs)
            data_sample.gt_instances.ignored = torch.BoolTensor([False] *
                                                                num_boxes)
            data_samples.append(data_sample)

            # kernels = []
            # TODO: add support for multiple kernels (if necessary)
            # for _ in range(num_kernels):
            #     kernel = np.random.rand(H, W)
            #     kernels.append(kernel)
            gt_kernels.append(np.random.rand(H, W))
            gt_effective_mask.append(np.ones((H, W)))

        mask = np.random.randint(0, 2, (len(boxes), H, W), dtype=np.uint8)
        gt_masks.append(mask)

        mm_inputs = {
            'imgs': torch.FloatTensor(imgs).requires_grad_(True),
            'data_samples': data_samples,
            'gt_masks': gt_masks,
            'gt_kernels': gt_kernels,
            'gt_mask': gt_effective_mask,
            'gt_thr_mask': gt_effective_mask,
            'gt_text_mask': gt_effective_mask,
            'gt_center_region_mask': gt_effective_mask,
            'gt_radius_map': gt_kernels,
            'gt_sin_map': gt_kernels,
            'gt_cos_map': gt_kernels,
        }
        return mm_inputs

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

    def _get_detector_cfg(self, fname: str) -> 'ConfigDict':
        """Grab configs necessary to create a detector.

        These are deep copied to allow for safe modification of parameters
        without influencing other tests.
        """
        config = self._get_config_module(fname)
        model = copy.deepcopy(config.model)
        return model
