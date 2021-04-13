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
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
    } for _ in range(N)]
    relations = [torch.randn(10, 10, 5) for _ in range(N)]
    texts = [torch.ones(10, 16) for _ in range(N)]
    gt_bboxes = [torch.Tensor([[2, 2, 4, 4]]).expand(10, 4) for _ in range(N)]
    gt_labels = [torch.ones(10, 11).long() for _ in range(N)]

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'img_metas': img_metas,
        'relations': relations,
        'texts': texts,
        'gt_bboxes': gt_bboxes,
        'gt_labels': gt_labels
    }
    return mm_inputs


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
    config.model.class_list = None
    model = copy.deepcopy(config.model)
    return model


@pytest.mark.parametrize('cfg_file', [
    'kie/sdmgr/sdmgr_novisual_60e_wildreceipt.py',
    'kie/sdmgr/sdmgr_unet16_60e_wildreceipt.py'
])
def test_sdmgr_pipeline(cfg_file):
    model = _get_detector_cfg(cfg_file)
    model['pretrained'] = None

    from mmocr.models import build_detector
    detector = build_detector(model)

    input_shape = (1, 3, 128, 128)

    mm_inputs = _demo_mm_inputs(0, input_shape)

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    relations = mm_inputs.pop('relations')
    texts = mm_inputs.pop('texts')
    gt_bboxes = mm_inputs.pop('gt_bboxes')
    gt_labels = mm_inputs.pop('gt_labels')

    # Test forward train
    losses = detector.forward(
        imgs,
        img_metas,
        relations=relations,
        texts=texts,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        batch_results = []
        for idx in range(len(img_metas)):
            result = detector.forward(
                imgs[idx:idx + 1],
                None,
                return_loss=False,
                relations=[relations[idx]],
                texts=[texts[idx]],
                gt_bboxes=[gt_bboxes[idx]])
            batch_results.append(result)

    # Test show_result
    results = {'nodes': torch.randn(1, 3)}
    boxes = [[1, 1, 2, 1, 2, 2, 1, 2]]
    img = np.random.rand(5, 5, 3)
    detector.show_result(img, results, boxes)
