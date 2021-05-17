"""pytest tests/test_detector.py."""
import copy
from os.path import dirname, exists, join

import numpy as np
import pytest
import torch

import mmocr.core.evaluation.utils as utils


def _demo_mm_inputs(num_kernels=0, input_shape=(1, 3, 300, 300),
                    num_items=None, num_classes=1):  # yapf: disable
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple): Input batch dimensions.

        num_items (None | list[int]): Specifies the number of boxes
            for each batch item.

        num_classes (int): Number of distinct labels a box might have.
    """
    from mmdet.core import BitmapMasks

    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)

    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': np.array([1, 1, 1, 1]),
        'flip': False,
    } for _ in range(N)]

    gt_bboxes = []
    gt_labels = []
    gt_masks = []
    gt_kernels = []
    gt_effective_mask = []

    for batch_idx in range(N):
        if num_items is None:
            num_boxes = rng.randint(1, 10)
        else:
            num_boxes = num_items[batch_idx]

        cx, cy, bw, bh = rng.rand(num_boxes, 4).T

        tl_x = ((cx * W) - (W * bw / 2)).clip(0, W)
        tl_y = ((cy * H) - (H * bh / 2)).clip(0, H)
        br_x = ((cx * W) + (W * bw / 2)).clip(0, W)
        br_y = ((cy * H) + (H * bh / 2)).clip(0, H)

        boxes = np.vstack([tl_x, tl_y, br_x, br_y]).T
        class_idxs = [0] * num_boxes

        gt_bboxes.append(torch.FloatTensor(boxes))
        gt_labels.append(torch.LongTensor(class_idxs))
        kernels = []
        for kernel_inx in range(num_kernels):
            kernel = np.random.rand(H, W)
            kernels.append(kernel)
        gt_kernels.append(BitmapMasks(kernels, H, W))
        gt_effective_mask.append(BitmapMasks([np.ones((H, W))], H, W))

    mask = np.random.randint(0, 2, (len(boxes), H, W), dtype=np.uint8)
    gt_masks.append(BitmapMasks(mask, H, W))

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'img_metas': img_metas,
        'gt_bboxes': gt_bboxes,
        'gt_labels': gt_labels,
        'gt_bboxes_ignore': None,
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
    'textdet/maskrcnn/mask_rcnn_r50_fpn_160e_ctw1500.py',
    'textdet/maskrcnn/mask_rcnn_r50_fpn_160e_icdar2015.py',
    'textdet/maskrcnn/mask_rcnn_r50_fpn_160e_icdar2017.py'
])
def test_ocr_mask_rcnn(cfg_file):
    model = _get_detector_cfg(cfg_file)
    model['pretrained'] = None

    from mmocr.models import build_detector
    detector = build_detector(model)

    input_shape = (1, 3, 224, 224)
    mm_inputs = _demo_mm_inputs(0, input_shape)

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_labels = mm_inputs.pop('gt_labels')
    gt_masks = mm_inputs.pop('gt_masks')

    # Test forward train
    gt_bboxes = mm_inputs['gt_bboxes']
    losses = detector.forward(
        imgs,
        img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        gt_masks=gt_masks)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]],
                                      return_loss=False)
            batch_results.append(result)

    # Test get_boundary
    results = ([[[1]]], [[
        np.array([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    ]])

    boundaries = detector.get_boundary(results)
    assert utils.boundary_iou(boundaries['boundary_result'][0][:-1],
                              [1, 1, 0, 1, 0, 0, 1, 0]) == 1

    # Test show_result

    results = {'boundary_result': [[0, 0, 1, 0, 1, 1, 0, 1, 0.9]]}
    img = np.random.rand(5, 5)
    detector.show_result(img, results)


@pytest.mark.parametrize('cfg_file', [
    'textdet/panet/panet_r18_fpem_ffm_600e_ctw1500.py',
    'textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py',
    'textdet/panet/panet_r50_fpem_ffm_600e_icdar2017.py'
])
def test_panet(cfg_file):
    model = _get_detector_cfg(cfg_file)
    model['pretrained'] = None
    model['backbone']['norm_cfg']['type'] = 'BN'

    from mmocr.models import build_detector
    detector = build_detector(model)

    input_shape = (1, 3, 224, 224)
    num_kernels = 2
    mm_inputs = _demo_mm_inputs(num_kernels, input_shape)

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_kernels = mm_inputs.pop('gt_kernels')
    gt_mask = mm_inputs.pop('gt_mask')

    # Test forward train
    losses = detector.forward(
        imgs, img_metas, gt_kernels=gt_kernels, gt_mask=gt_mask)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]],
                                      return_loss=False)
            batch_results.append(result)

    # Test show result
    results = {'boundary_result': [[0, 0, 1, 0, 1, 1, 0, 1, 0.9]]}
    img = np.random.rand(5, 5)
    detector.show_result(img, results)


@pytest.mark.parametrize('cfg_file', [
    'textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py',
    'textdet/psenet/psenet_r50_fpnf_600e_icdar2017.py',
    'textdet/psenet/psenet_r50_fpnf_600e_ctw1500.py'
])
def test_psenet(cfg_file):
    model = _get_detector_cfg(cfg_file)
    model['pretrained'] = None
    model['backbone']['norm_cfg']['type'] = 'BN'

    from mmocr.models import build_detector
    detector = build_detector(model)

    input_shape = (1, 3, 224, 224)
    num_kernels = 7
    mm_inputs = _demo_mm_inputs(num_kernels, input_shape)

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_kernels = mm_inputs.pop('gt_kernels')
    gt_mask = mm_inputs.pop('gt_mask')

    # Test forward train
    losses = detector.forward(
        imgs, img_metas, gt_kernels=gt_kernels, gt_mask=gt_mask)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]],
                                      return_loss=False)
            batch_results.append(result)

    # Test show result
    results = {'boundary_result': [[0, 0, 1, 0, 1, 1, 0, 1, 0.9]]}
    img = np.random.rand(5, 5)
    detector.show_result(img, results)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
@pytest.mark.parametrize('cfg_file', [
    'textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py',
    'textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py'
])
def test_dbnet(cfg_file):
    model = _get_detector_cfg(cfg_file)
    model['pretrained'] = None
    model['backbone']['norm_cfg']['type'] = 'BN'

    from mmocr.models import build_detector
    detector = build_detector(model)
    detector = detector.cuda()
    input_shape = (1, 3, 224, 224)
    num_kernels = 7
    mm_inputs = _demo_mm_inputs(num_kernels, input_shape)

    imgs = mm_inputs.pop('imgs')
    imgs = imgs.cuda()
    img_metas = mm_inputs.pop('img_metas')
    gt_shrink = mm_inputs.pop('gt_kernels')
    gt_shrink_mask = mm_inputs.pop('gt_mask')
    gt_thr = mm_inputs.pop('gt_masks')
    gt_thr_mask = mm_inputs.pop('gt_thr_mask')

    # Test forward train
    losses = detector.forward(
        imgs,
        img_metas,
        gt_shrink=gt_shrink,
        gt_shrink_mask=gt_shrink_mask,
        gt_thr=gt_thr,
        gt_thr_mask=gt_thr_mask)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]],
                                      return_loss=False)
            batch_results.append(result)

    # Test show result
    results = {'boundary_result': [[0, 0, 1, 0, 1, 1, 0, 1, 0.9]]}
    img = np.random.rand(5, 5)
    detector.show_result(img, results)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
@pytest.mark.parametrize(
    'cfg_file',
    ['textdet/textsnake/'
     'textsnake_r50_fpn_unet_1200e_ctw1500.py'])
def test_textsnake(cfg_file):
    model = _get_detector_cfg(cfg_file)
    model['pretrained'] = None
    model['backbone']['norm_cfg']['type'] = 'BN'

    from mmocr.models import build_detector
    detector = build_detector(model)
    detector = detector.cuda()
    input_shape = (1, 3, 224, 224)
    num_kernels = 1
    mm_inputs = _demo_mm_inputs(num_kernels, input_shape)

    imgs = mm_inputs.pop('imgs')
    imgs = imgs.cuda()
    img_metas = mm_inputs.pop('img_metas')
    gt_text_mask = mm_inputs.pop('gt_text_mask')
    gt_center_region_mask = mm_inputs.pop('gt_center_region_mask')
    gt_mask = mm_inputs.pop('gt_mask')
    gt_radius_map = mm_inputs.pop('gt_radius_map')
    gt_sin_map = mm_inputs.pop('gt_sin_map')
    gt_cos_map = mm_inputs.pop('gt_cos_map')

    # Test forward train
    losses = detector.forward(
        imgs,
        img_metas,
        gt_text_mask=gt_text_mask,
        gt_center_region_mask=gt_center_region_mask,
        gt_mask=gt_mask,
        gt_radius_map=gt_radius_map,
        gt_sin_map=gt_sin_map,
        gt_cos_map=gt_cos_map)
    assert isinstance(losses, dict)

    # Test forward test get_boundary
    maps = torch.zeros((1, 5, 224, 224), dtype=torch.float)
    maps[:, 0:2, :, :] = -10.
    maps[:, 0, 60:100, 12:212] = 10.
    maps[:, 1, 70:90, 22:202] = 10.
    maps[:, 2, 70:90, 22:202] = 0.
    maps[:, 3, 70:90, 22:202] = 1.
    maps[:, 4, 70:90, 22:202] = 10.

    one_meta = img_metas[0]
    result = detector.bbox_head.get_boundary(maps, [one_meta], False)
    assert len(result) == 1

    # Test show result
    results = {'boundary_result': [[0, 0, 1, 0, 1, 1, 0, 1, 0.9]]}
    img = np.random.rand(5, 5)
    detector.show_result(img, results)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
@pytest.mark.parametrize(
    'cfg_file', ['textdet/fcenet/fcenet_r50dcnv2_fpn_1500e_ctw1500.py'])
def test_fcenet(cfg_file):
    model = _get_detector_cfg(cfg_file)
    model['pretrained'] = None
    model['backbone']['norm_cfg']['type'] = 'BN'

    from mmocr.models import build_detector
    detector = build_detector(model)
    detector = detector.cuda()

    fourier_degree = 5
    input_shape = (1, 3, 256, 256)
    (n, c, h, w) = input_shape

    imgs = torch.randn(n, c, h, w).float().cuda()
    img_metas = [{
        'img_shape': (h, w, c),
        'ori_shape': (h, w, c),
        'pad_shape': (h, w, c),
        'filename': '<demo>.png',
        'scale_factor': np.array([1, 1, 1, 1]),
        'flip': False,
    } for _ in range(n)]

    p3_maps = []
    p4_maps = []
    p5_maps = []
    for _ in range(n):
        p3_maps.append(
            np.random.random((5 + 4 * fourier_degree, h // 8, w // 8)))
        p4_maps.append(
            np.random.random((5 + 4 * fourier_degree, h // 16, w // 16)))
        p5_maps.append(
            np.random.random((5 + 4 * fourier_degree, h // 32, w // 32)))

    # Test forward train
    losses = detector.forward(
        imgs, img_metas, p3_maps=p3_maps, p4_maps=p4_maps, p5_maps=p5_maps)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]],
                                      return_loss=False)
            batch_results.append(result)

    # Test show result
    results = {'boundary_result': [[0, 0, 1, 0, 1, 1, 0, 1, 0.9]]}
    img = np.random.rand(5, 5)
    detector.show_result(img, results)


@pytest.mark.parametrize(
    'cfg_file', ['textdet/drrg/'
                 'drrg_r50_fpn_unet_1200e_ctw1500.py'])
def test_drrg(cfg_file):
    model = _get_detector_cfg(cfg_file)
    model['pretrained'] = None
    model['backbone']['norm_cfg']['type'] = 'BN'

    from mmocr.models import build_detector
    detector = build_detector(model)

    input_shape = (1, 3, 224, 224)
    num_kernels = 1
    mm_inputs = _demo_mm_inputs(num_kernels, input_shape)

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_text_mask = mm_inputs.pop('gt_text_mask')
    gt_center_region_mask = mm_inputs.pop('gt_center_region_mask')
    gt_mask = mm_inputs.pop('gt_mask')
    gt_top_height_map = mm_inputs.pop('gt_radius_map')
    gt_bot_height_map = gt_top_height_map.copy()
    gt_sin_map = mm_inputs.pop('gt_sin_map')
    gt_cos_map = mm_inputs.pop('gt_cos_map')
    roi_num = 32
    x = np.random.randint(4, 224, (roi_num, 1))
    y = np.random.randint(4, 224, (roi_num, 1))
    h = 4 * np.ones((roi_num, 1))
    w = 4 * np.ones((roi_num, 1))
    angle = (np.random.random_sample((roi_num, 1)) * 2 - 1) * np.pi / 2
    cos, sin = np.cos(angle), np.sin(angle)
    comp_labels = np.random.randint(1, 3, (roi_num, 1))
    roi_num = roi_num * np.ones((roi_num, 1))
    comp_attribs = np.hstack([roi_num, x, y, h, w, cos, sin, comp_labels])
    gt_comp_attribs = np.expand_dims(comp_attribs.astype(np.float32), axis=0)

    # Test forward train
    losses = detector.forward(
        imgs,
        img_metas,
        gt_text_mask=gt_text_mask,
        gt_center_region_mask=gt_center_region_mask,
        gt_mask=gt_mask,
        gt_top_height_map=gt_top_height_map,
        gt_bot_height_map=gt_bot_height_map,
        gt_sin_map=gt_sin_map,
        gt_cos_map=gt_cos_map,
        gt_comp_attribs=gt_comp_attribs)
    assert isinstance(losses, dict)

    # Test forward test
    input_shape = (1, 3, 64, 64)
    num_kernels = 1
    mm_inputs = _demo_mm_inputs(num_kernels, input_shape)
    imgs = mm_inputs.pop('imgs')

    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]],
                                      return_loss=False)
            batch_results.append(result)

    # Test show result
    results = {'boundary_result': [[0, 0, 1, 0, 1, 1, 0, 1, 0.9]]}
    img = np.random.rand(5, 5)
    detector.show_result(img, results)
