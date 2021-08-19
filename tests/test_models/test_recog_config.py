# Copyright (c) OpenMMLab. All rights reserved.
import copy
from os.path import dirname, exists, join

import numpy as np
import pytest
import torch


def _demo_mm_inputs(num_kernels=0, input_shape=(1, 3, 300, 300),
                    num_items=None):  # yapf: disable
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple): Input batch dimensions.

        num_items (None | list[int]): Specifies the number of boxes
            for each batch item.
    """

    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)

    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'resize_shape': (H, W, C),
        'filename': '<demo>.png',
        'text': 'hello',
        'valid_ratio': 1.0,
    } for _ in range(N)]

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'img_metas': img_metas
    }
    return mm_inputs


def _demo_gt_kernel_inputs(num_kernels=3, input_shape=(1, 3, 300, 300),
                           num_items=None):  # yapf: disable
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple): Input batch dimensions.

        num_items (None | list[int]): Specifies the number of boxes
            for each batch item.
    """
    from mmdet.core import BitmapMasks

    (N, C, H, W) = input_shape
    gt_kernels = []

    for batch_idx in range(N):
        kernels = []
        for kernel_inx in range(num_kernels):
            kernel = np.random.rand(H, W)
            kernels.append(kernel)
        gt_kernels.append(BitmapMasks(kernels, H, W))

    return gt_kernels


def _get_config_directory():
    """Find the predefined detector config directory."""
    try:
        # Assume we are running in the source mmocr repo
        repo_dpath = dirname(dirname(dirname(__file__)))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmocr
        repo_dpath = dirname(dirname(mmocr.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmcv import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod


def _get_detector_cfg(fname):
    """Grab configs necessary to create a detector.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    return model


@pytest.mark.parametrize('cfg_file', [
    'textrecog/sar/sar_r31_parallel_decoder_academic.py',
    'textrecog/crnn/crnn_academic_dataset.py',
    'textrecog/nrtr/nrtr_r31_1by16_1by8_academic.py',
    'textrecog/robust_scanner/robustscanner_r31_academic.py',
    'textrecog/seg/seg_r31_1by16_fpnocr_academic.py',
    'textrecog/satrn/satrn_academic.py'
])
def test_recognizer_pipeline(cfg_file):
    model = _get_detector_cfg(cfg_file)
    model['pretrained'] = None

    from mmocr.models import build_detector
    detector = build_detector(model)

    input_shape = (1, 3, 32, 160)
    if 'crnn' in cfg_file:
        input_shape = (1, 1, 32, 160)
    mm_inputs = _demo_mm_inputs(0, input_shape)
    gt_kernels = None
    if 'seg' in cfg_file:
        gt_kernels = _demo_gt_kernel_inputs(3, input_shape)

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')

    # Test forward train
    if 'seg' in cfg_file:
        losses = detector.forward(imgs, img_metas, gt_kernels=gt_kernels)
    else:
        losses = detector.forward(imgs, img_metas)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]],
                                      return_loss=False)
            batch_results.append(result)

    # Test show_result

    results = {'text': 'hello', 'score': 1.0}
    img = np.random.rand(5, 5, 3)
    detector.show_result(img, results)
