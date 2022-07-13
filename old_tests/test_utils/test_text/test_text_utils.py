# Copyright (c) OpenMMLab. All rights reserved.
"""Test text label visualize."""
import os.path as osp
import random
import tempfile
from unittest import mock

import numpy as np
import pytest

import mmocr.visualization.visualize as visualize_utils


def test_tile_image():
    dummp_imgs, heights, widths = [], [], []
    for _ in range(3):
        h = random.randint(100, 300)
        w = random.randint(100, 300)
        heights.append(h)
        widths.append(w)
        # dummy_img = Image.new('RGB', (w, h), Image.ANTIALIAS)
        dummy_img = np.ones((h, w, 3), dtype=np.uint8)
        dummp_imgs.append(dummy_img)
    joint_img = visualize_utils.tile_image(dummp_imgs)
    assert joint_img.shape[0] == sum(heights)
    assert joint_img.shape[1] == max(widths)

    # test invalid arguments
    with pytest.raises(AssertionError):
        visualize_utils.tile_image(dummp_imgs[0])
    with pytest.raises(AssertionError):
        visualize_utils.tile_image([])


@mock.patch('%s.visualize_utils.mmcv.imread' % __name__)
@mock.patch('%s.visualize_utils.mmcv.imshow' % __name__)
@mock.patch('%s.visualize_utils.mmcv.imwrite' % __name__)
def test_show_text_label(mock_imwrite, mock_imshow, mock_imread):
    img = np.ones((32, 160), dtype=np.uint8)
    pred_label = 'hello'
    gt_label = 'world'

    tmp_dir = tempfile.TemporaryDirectory()
    out_file = osp.join(tmp_dir.name, 'tmp.jpg')

    # test invalid arguments
    with pytest.raises(AssertionError):
        visualize_utils.imshow_text_label(5, pred_label, gt_label)
    with pytest.raises(AssertionError):
        visualize_utils.imshow_text_label(img, pred_label, 4)
    with pytest.raises(AssertionError):
        visualize_utils.imshow_text_label(img, 3, gt_label)
    with pytest.raises(AssertionError):
        visualize_utils.imshow_text_label(
            img, pred_label, gt_label, show=True, wait_time=0.1)

    mock_imread.side_effect = [img, img]
    visualize_utils.imshow_text_label(
        img, pred_label, gt_label, out_file=out_file)
    visualize_utils.imshow_text_label(
        img, '中文', '中文', out_file=None, show=True)

    # test showing img
    mock_imshow.assert_called_once()
    mock_imwrite.assert_called_once()

    tmp_dir.cleanup()
