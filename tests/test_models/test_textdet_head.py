import numpy as np
import torch

from mmocr.models.textdet.dense_heads import DRRGHead


def test_drrg_head():
    in_channels = 10
    drrg_head = DRRGHead(in_channels)
    assert drrg_head.in_channels == in_channels
    assert drrg_head.k_at_hops == (8, 4)
    assert drrg_head.adjacent_linkage_num == 3
    assert drrg_head.node_geo_feat_len == 120
    assert np.allclose(drrg_head.pooling_scale, 1.0)
    assert drrg_head.pooling_output_size == (4, 3)
    assert np.allclose(drrg_head.nms_thr, 0.3)
    assert np.allclose(drrg_head.min_width, 8.0)
    assert np.allclose(drrg_head.max_width, 24.0)
    assert np.allclose(drrg_head.comp_shrink_ratio, 1.03)
    assert np.allclose(drrg_head.comp_ratio, 0.4)
    assert np.allclose(drrg_head.comp_score_thr, 0.3)
    assert np.allclose(drrg_head.text_region_thr, 0.2)
    assert np.allclose(drrg_head.center_region_thr, 0.2)
    assert drrg_head.center_region_area_thr == 50
    assert np.allclose(drrg_head.local_graph_thr, 0.7)
    assert np.allclose(drrg_head.link_thr, 0.85)

    # test forward train
    roi_num = 16
    feature_maps = torch.randn((2, 10, 128, 128), dtype=torch.float)
    x = np.random.randint(4, 124, (roi_num, 1))
    y = np.random.randint(4, 124, (roi_num, 1))
    h = 4 * np.ones((roi_num, 1))
    w = 4 * np.ones((roi_num, 1))
    angle = (np.random.random_sample((roi_num, 1)) * 2 - 1) * np.pi / 2
    cos, sin = np.cos(angle), np.sin(angle)
    comp_labels = np.random.randint(1, 3, (roi_num, 1))
    roi_num = roi_num * np.ones((roi_num, 1))
    comp_attribs = np.hstack([roi_num, x, y, h, w, cos, sin, comp_labels])
    comp_attribs = comp_attribs.astype(np.float32)
    comp_attribs_ = comp_attribs.copy()
    comp_attribs = np.stack([comp_attribs, comp_attribs_])
    pred_maps, gcn_data = drrg_head(feature_maps, comp_attribs)
    pred_labels, gt_labels = gcn_data
    assert pred_maps.size() == (2, 6, 128, 128)
    assert pred_labels.ndim == gt_labels.ndim == 2
    assert gt_labels.size()[0] * gt_labels.size()[1] == pred_labels.size()[0]
    assert pred_labels.size()[1] == 2

    # test get_boundary
    edges = np.stack([np.arange(0, 9), np.arange(1, 10)]).transpose()
    scores = np.ones(9, dtype=np.float32) * 0.9
    x1 = np.arange(2, 22, 2)
    x2 = x1 + 2
    y1 = np.ones(10) * 2
    y2 = y1 + 2
    comp_scores = np.ones(10, dtype=np.float32) * 0.9
    text_comps = np.stack([x1, y1, x2, y1, x2, y2, x1, y2, 
                           comp_scores]).transpose()

    (C, H, W) = (10, 128, 128)
    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': np.array([1, 1, 1, 1]),
        'flip': False,
    }]
    results = drrg_head.get_boundary(
        edges, scores, text_comps, img_metas, rescale=True)
    assert 'boundary_result' in results.keys()
