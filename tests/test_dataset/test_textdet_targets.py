from unittest import mock

import numpy as np

import mmocr.datasets.pipelines.custom_format_bundle as cf_bundle
import mmocr.datasets.pipelines.textdet_targets as textdet_targets
from mmdet.core import PolygonMasks


@mock.patch('%s.cf_bundle.show_feature' % __name__)
def test_gen_pannet_targets(mock_show_feature):

    target_generator = textdet_targets.PANetTargets()
    assert target_generator.max_shrink == 20

    # test generate_kernels
    img_size = (3, 10)
    text_polys = [[np.array([0, 0, 1, 0, 1, 1, 0, 1])],
                  [np.array([2, 0, 3, 0, 3, 1, 2, 1])]]
    shrink_ratio = 1.0
    kernel = np.array([[1, 1, 2, 2, 0, 0, 0, 0, 0, 0],
                       [1, 1, 2, 2, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    output, _ = target_generator.generate_kernels(img_size, text_polys,
                                                  shrink_ratio)
    print(output)
    assert np.allclose(output, kernel)

    # test generate_effective_mask
    polys_ignore = text_polys
    output = target_generator.generate_effective_mask((3, 10), polys_ignore)
    target = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

    assert np.allclose(output, target)

    # test generate_targets
    results = {}
    results['img'] = np.zeros((3, 10, 3), np.uint8)
    results['gt_masks'] = PolygonMasks(text_polys, 3, 10)
    results['gt_masks_ignore'] = PolygonMasks([], 3, 10)
    results['img_shape'] = (3, 10, 3)
    results['mask_fields'] = []
    output = target_generator(results)
    assert len(output['gt_kernels']) == 2
    assert len(output['gt_mask']) == 1

    bundle = cf_bundle.CustomFormatBundle(
        keys=['gt_kernels', 'gt_mask'],
        visualize=dict(flag=True, boundary_key='gt_kernels'))
    bundle(output)
    assert 'gt_kernels' in output.keys()
    assert 'gt_mask' in output.keys()
    mock_show_feature.assert_called_once()


def test_gen_psenet_targets():
    target_generator = textdet_targets.PSENetTargets()
    assert target_generator.max_shrink == 20
    assert target_generator.shrink_ratio == (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)


# Test DBNetTargets


def test_dbnet_targets_find_invalid():
    target_generator = textdet_targets.DBNetTargets()
    assert target_generator.shrink_ratio == 0.4
    assert target_generator.thr_min == 0.3
    assert target_generator.thr_max == 0.7

    results = {}
    text_polys = [[np.array([0, 0, 10, 0, 10, 10, 0, 10])],
                  [np.array([20, 0, 30, 0, 30, 10, 20, 10])]]
    results['gt_masks'] = PolygonMasks(text_polys, 40, 40)

    ignore_tags = target_generator.find_invalid(results)
    assert np.allclose(ignore_tags, [False, False])


def test_dbnet_targets():
    target_generator = textdet_targets.DBNetTargets()
    assert target_generator.shrink_ratio == 0.4
    assert target_generator.thr_min == 0.3
    assert target_generator.thr_max == 0.7


def test_dbnet_ignore_texts():
    target_generator = textdet_targets.DBNetTargets()
    ignore_tags = [True, False]
    results = {}
    text_polys = [[np.array([0, 0, 10, 0, 10, 10, 0, 10])],
                  [np.array([20, 0, 30, 0, 30, 10, 20, 10])]]
    text_polys_ignore = [[np.array([0, 0, 15, 0, 15, 10, 0, 10])]]

    results['gt_masks_ignore'] = PolygonMasks(text_polys_ignore, 40, 40)
    results['gt_masks'] = PolygonMasks(text_polys, 40, 40)
    results['gt_bboxes'] = np.array([[0, 0, 10, 10], [20, 0, 30, 10]])
    results['gt_labels'] = np.array([0, 1])

    target_generator.ignore_texts(results, ignore_tags)

    assert np.allclose(results['gt_labels'], np.array([1]))
    assert len(results['gt_masks_ignore'].masks) == 2
    assert np.allclose(results['gt_masks_ignore'].masks[1][0],
                       text_polys[0][0])
    assert len(results['gt_masks'].masks) == 1


def test_dbnet_generate_thr_map():
    target_generator = textdet_targets.DBNetTargets()
    text_polys = [[np.array([0, 0, 10, 0, 10, 10, 0, 10])],
                  [np.array([20, 0, 30, 0, 30, 10, 20, 10])]]
    thr_map, thr_mask = target_generator.generate_thr_map((40, 40), text_polys)
    assert np.all((thr_map >= 0.29) * (thr_map <= 0.71))


def test_dbnet_generate_targets():
    target_generator = textdet_targets.DBNetTargets()
    text_polys = [[np.array([0, 0, 10, 0, 10, 10, 0, 10])],
                  [np.array([20, 0, 30, 0, 30, 10, 20, 10])]]
    text_polys_ignore = [[np.array([0, 0, 15, 0, 15, 10, 0, 10])]]

    results = {}
    results['mask_fields'] = []
    results['img_shape'] = (40, 40, 3)
    results['gt_masks_ignore'] = PolygonMasks(text_polys_ignore, 40, 40)
    results['gt_masks'] = PolygonMasks(text_polys, 40, 40)
    results['gt_bboxes'] = np.array([[0, 0, 10, 10], [20, 0, 30, 10]])
    results['gt_labels'] = np.array([0, 1])

    target_generator.generate_targets(results)
    assert 'gt_shrink' in results['mask_fields']
    assert 'gt_shrink_mask' in results['mask_fields']
    assert 'gt_thr' in results['mask_fields']
    assert 'gt_thr_mask' in results['mask_fields']


@mock.patch('%s.cf_bundle.show_feature' % __name__)
def test_gen_textsnake_targets(mock_show_feature):

    target_generator = textdet_targets.TextSnakeTargets()
    assert np.allclose(target_generator.orientation_thr, 2.0)
    assert np.allclose(target_generator.resample_step, 4.0)
    assert np.allclose(target_generator.center_region_shrink_ratio, 0.3)

    # test find_head_tail for quadrangle
    polygon = np.array([[1.0, 1.0], [5.0, 1.0], [5.0, 3.0], [1.0, 3.0]])
    head_inds, tail_inds = target_generator.find_head_tail(polygon, 2.0)
    assert np.allclose(head_inds, [3, 0])
    assert np.allclose(tail_inds, [1, 2])

    # test find_head_tail for polygon
    polygon = np.array([[0., 10.], [3., 3.], [10., 0.], [17., 3.], [20., 10.],
                        [15., 10.], [13.5, 6.5], [10., 5.], [6.5, 6.5],
                        [5., 10.]])
    head_inds, tail_inds = target_generator.find_head_tail(polygon, 2.0)
    assert np.allclose(head_inds, [9, 0])
    assert np.allclose(tail_inds, [4, 5])

    # test generate_text_region_mask
    img_size = (3, 10)
    text_polys = [[np.array([0, 0, 1, 0, 1, 1, 0, 1])],
                  [np.array([2, 0, 3, 0, 3, 1, 2, 1])]]
    output = target_generator.generate_text_region_mask(img_size, text_polys)
    target = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert np.allclose(output, target)

    # test generate_center_region_mask
    target_generator.center_region_shrink_ratio = 1.0
    (center_region_mask, radius_map, sin_map,
     cos_map) = target_generator.generate_center_mask_attrib_maps(
         img_size, text_polys)
    target = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert np.allclose(center_region_mask, target)
    assert np.allclose(sin_map, np.zeros(img_size))
    assert np.allclose(cos_map, target)

    # test generate_effective_mask
    polys_ignore = text_polys
    output = target_generator.generate_effective_mask(img_size, polys_ignore)
    target = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    assert np.allclose(output, target)

    # test generate_targets
    results = {}
    results['img'] = np.zeros((3, 10, 3), np.uint8)
    results['gt_masks'] = PolygonMasks(text_polys, 3, 10)
    results['gt_masks_ignore'] = PolygonMasks([], 3, 10)
    results['img_shape'] = (3, 10, 3)
    results['mask_fields'] = []
    output = target_generator(results)
    assert len(output['gt_text_mask']) == 1
    assert len(output['gt_center_region_mask']) == 1
    assert len(output['gt_mask']) == 1
    assert len(output['gt_radius_map']) == 1
    assert len(output['gt_sin_map']) == 1
    assert len(output['gt_cos_map']) == 1

    bundle = cf_bundle.CustomFormatBundle(
        keys=[
            'gt_text_mask', 'gt_center_region_mask', 'gt_mask',
            'gt_radius_map', 'gt_sin_map', 'gt_cos_map'
        ],
        visualize=dict(flag=True, boundary_key='gt_text_mask'))
    bundle(output)
    assert 'gt_text_mask' in output.keys()
    assert 'gt_center_region_mask' in output.keys()
    assert 'gt_mask' in output.keys()
    assert 'gt_radius_map' in output.keys()
    assert 'gt_sin_map' in output.keys()
    assert 'gt_cos_map' in output.keys()
    mock_show_feature.assert_called_once()


def test_fcenet_generate_targets():
    fourier_degree = 5
    target_generator = textdet_targets.FCENetTargets(
        fourier_degree=fourier_degree)

    h, w, c = (64, 64, 3)
    text_polys = [[np.array([0, 0, 10, 0, 10, 10, 0, 10])],
                  [np.array([20, 0, 30, 0, 30, 10, 20, 10])]]
    text_polys_ignore = [[np.array([0, 0, 15, 0, 15, 10, 0, 10])]]

    results = {}
    results['mask_fields'] = []
    results['img_shape'] = (h, w, c)
    results['gt_masks_ignore'] = PolygonMasks(text_polys_ignore, h, w)
    results['gt_masks'] = PolygonMasks(text_polys, h, w)
    results['gt_bboxes'] = np.array([[0, 0, 10, 10], [20, 0, 30, 10]])
    results['gt_labels'] = np.array([0, 1])

    target_generator.generate_targets(results)
    assert 'p3_maps' in results.keys()
    assert 'p4_maps' in results.keys()
    assert 'p5_maps' in results.keys()


def test_gen_drrg_targets():
    target_generator = textdet_targets.DRRGTargets()
    assert np.allclose(target_generator.orientation_thr, 2.0)
    assert np.allclose(target_generator.resample_step, 8.0)
    assert target_generator.min_comp_num == 9
    assert target_generator.max_comp_num == 600
    assert np.allclose(target_generator.min_width, 8.0)
    assert np.allclose(target_generator.max_width, 24.0)
    assert np.allclose(target_generator.center_region_shrink_ratio, 0.3)
    assert np.allclose(target_generator.comp_shrink_ratio, 1.0)
    assert np.allclose(target_generator.text_comp_ratio, 0.3)
    assert np.allclose(target_generator.text_comp_nms_thr, 0.25)
    assert np.allclose(target_generator.min_rand_half_height, 8.0)
    assert np.allclose(target_generator.max_rand_half_height, 24.0)
    assert np.allclose(target_generator.jitter_level, 0.2)

    # test generate_targets
    target_generator = textdet_targets.DRRGTargets(
        min_width=2.,
        max_width=4.,
        min_rand_half_height=3.,
        max_rand_half_height=5.)

    results = {}
    results['img'] = np.zeros((20, 30, 3), np.uint8)
    text_polys = [[np.array([4, 2, 18, 2, 18, 10, 4, 10])],
                  [np.array([8, 12, 8, 12, 23, 18, 23, 18])]]
    results['gt_masks'] = PolygonMasks(text_polys, 20, 30)
    results['gt_masks_ignore'] = PolygonMasks([], 20, 30)
    results['img_shape'] = (20, 30, 3)
    results['mask_fields'] = []
    output = target_generator(results)
    assert len(output['gt_text_mask']) == 1
    assert len(output['gt_center_region_mask']) == 1
    assert len(output['gt_mask']) == 1
    assert len(output['gt_top_height_map']) == 1
    assert len(output['gt_bot_height_map']) == 1
    assert len(output['gt_sin_map']) == 1
    assert len(output['gt_cos_map']) == 1
    assert output['gt_comp_attribs'].shape[-1] == 8

    # test generate_targets with blank polygon masks
    results = {}
    results['img'] = np.zeros((20, 30, 3), np.uint8)
    results['gt_masks'] = PolygonMasks([], 20, 30)
    results['gt_masks_ignore'] = PolygonMasks([], 20, 30)
    results['img_shape'] = (20, 30, 3)
    results['mask_fields'] = []
    output = target_generator(results)
    assert len(output['gt_text_mask']) == 1
    assert len(output['gt_center_region_mask']) == 1
    assert len(output['gt_mask']) == 1
    assert len(output['gt_top_height_map']) == 1
    assert len(output['gt_bot_height_map']) == 1
    assert len(output['gt_sin_map']) == 1
    assert len(output['gt_cos_map']) == 1
    assert output['gt_comp_attribs'].shape[0] > 8
    assert output['gt_comp_attribs'].shape[-1] == 8
