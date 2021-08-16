# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

import mmocr.models.textdet.dense_heads.pan_head as pan_head


def test_panhead():
    in_channels = [128]
    out_channels = 128
    text_repr_type = 'poly'  # 'poly' or 'quad'
    downsample_ratio = 0.25
    loss = dict(type='PANLoss')

    # test invalid arguments
    with pytest.raises(AssertionError):
        panheader = pan_head.PANHead(128, out_channels, text_repr_type,
                                     downsample_ratio, loss)
    with pytest.raises(AssertionError):
        panheader = pan_head.PANHead(in_channels, [out_channels],
                                     text_repr_type, downsample_ratio, loss)
    with pytest.raises(AssertionError):
        panheader = pan_head.PANHead(in_channels, out_channels, 'test',
                                     text_repr_type, downsample_ratio, loss)
    with pytest.raises(AssertionError):
        panheader = pan_head.PANHead(in_channels, out_channels, 'test',
                                     downsample_ratio, loss)
    with pytest.raises(AssertionError):
        panheader = pan_head.PANHead(in_channels, out_channels, text_repr_type,
                                     1.1, loss)

    panheader = pan_head.PANHead(in_channels, out_channels, text_repr_type,
                                 downsample_ratio, loss)

    # test resize_boundary
    boundaries = [[0, 0, 0, 1, 1, 1, 0, 1, 0.9],
                  [0, 0, 0, 0.1, 0.1, 0.1, 0, 0.1, 0.9]]
    target_boundary = [[0, 0, 0, 0.5, 1, 0.5, 0, 0.5, 0.9],
                       [0, 0, 0, 0.05, 0.1, 0.05, 0, 0.05, 0.9]]
    scale_factor = np.array([1, 0.5, 1, 0.5])
    resized_boundary = panheader.resize_boundary(boundaries, scale_factor)
    assert np.allclose(resized_boundary, target_boundary)
