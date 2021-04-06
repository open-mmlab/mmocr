"""Test text mask_utils."""
import tempfile
from unittest import mock

import numpy as np
import pytest

import mmocr.core.evaluation.utils as eval_utils
import mmocr.core.mask as mask_utils
import mmocr.core.visualize as visualize_utils


def test_points2boundary():

    points = np.array([[1, 2]])
    text_repr_type = 'quad'
    text_score = None

    # test invalid arguments
    with pytest.raises(AssertionError):
        mask_utils.points2boundary([], text_repr_type, text_score)

    with pytest.raises(AssertionError):
        mask_utils.points2boundary(points, '', text_score)
    with pytest.raises(AssertionError):
        mask_utils.points2boundary(points, '', 1.1)

    # test quad
    points = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2],
                       [1, 2], [2, 2]])
    text_repr_type = 'quad'
    text_score = None

    result = mask_utils.points2boundary(points, text_repr_type, text_score)
    pred_poly = eval_utils.points2polygon(result)
    target_poly = eval_utils.points2polygon([2, 2, 0, 2, 0, 0, 2, 0])
    assert eval_utils.poly_iou(pred_poly, target_poly) == 1

    # test poly
    text_repr_type = 'poly'
    result = mask_utils.points2boundary(points, text_repr_type, text_score)
    pred_poly = eval_utils.points2polygon(result)
    target_poly = eval_utils.points2polygon([0, 0, 0, 2, 2, 2, 2, 0])
    assert eval_utils.poly_iou(pred_poly, target_poly) == 1


def test_seg2boundary():

    seg = np.array([[]])
    text_repr_type = 'quad'
    text_score = None
    # test invalid arguments
    with pytest.raises(AssertionError):
        mask_utils.seg2boundary([[]], text_repr_type, text_score)
    with pytest.raises(AssertionError):
        mask_utils.seg2boundary(seg, 1, text_score)
    with pytest.raises(AssertionError):
        mask_utils.seg2boundary(seg, text_repr_type, 1.1)

    seg = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    result = mask_utils.seg2boundary(seg, text_repr_type, text_score)
    pred_poly = eval_utils.points2polygon(result)
    target_poly = eval_utils.points2polygon([2, 2, 0, 2, 0, 0, 2, 0])
    assert eval_utils.poly_iou(pred_poly, target_poly) == 1


@mock.patch('%s.visualize_utils.plt' % __name__)
def test_show_feature(mock_plt):

    features = [np.random.rand(10, 10)]
    names = ['test']
    to_uint8 = [0]
    out_file = None

    # test invalid arguments
    with pytest.raises(AssertionError):
        visualize_utils.show_feature([], names, to_uint8, out_file)
    with pytest.raises(AssertionError):
        visualize_utils.show_feature(features, [1], to_uint8, out_file)
    with pytest.raises(AssertionError):
        visualize_utils.show_feature(features, names, ['a'], out_file)
    with pytest.raises(AssertionError):
        visualize_utils.show_feature(features, names, to_uint8, 1)
    with pytest.raises(AssertionError):
        visualize_utils.show_feature(features, names, to_uint8, [0, 1])

    visualize_utils.show_feature(features, names, to_uint8)

    # test showing img
    mock_plt.title.assert_called_once_with('test')
    mock_plt.show.assert_called_once()

    # test saving fig
    out_file = tempfile.NamedTemporaryFile().name
    visualize_utils.show_feature(features, names, to_uint8, out_file)
    mock_plt.savefig.assert_called_once()


@mock.patch('%s.visualize_utils.plt' % __name__)
def test_show_img_boundary(mock_plt):
    img = np.random.rand(10, 10)
    boundary = [0, 0, 1, 0, 1, 1, 0, 1]
    # test invalid arguments
    with pytest.raises(AssertionError):
        visualize_utils.show_img_boundary([], boundary)
    with pytest.raises(AssertionError):
        visualize_utils.show_img_boundary(img, np.array([]))

    # test showing img

    visualize_utils.show_img_boundary(img, boundary)
    mock_plt.imshow.assert_called_once()
    mock_plt.show.assert_called_once()


@mock.patch('%s.visualize_utils.mmcv' % __name__)
def test_show_pred_gt(mock_mmcv):
    preds = [[0, 0, 1, 0, 1, 1, 0, 1]]
    gts = [[0, 0, 1, 0, 1, 1, 0, 1]]
    show = True
    win_name = 'test'
    wait_time = 0
    out_file = tempfile.NamedTemporaryFile().name

    with pytest.raises(AssertionError):
        visualize_utils.show_pred_gt(np.array([]), gts)
    with pytest.raises(AssertionError):
        visualize_utils.show_pred_gt(preds, np.array([]))

    # test showing img

    visualize_utils.show_pred_gt(preds, gts, show, win_name, wait_time,
                                 out_file)
    mock_mmcv.imshow.assert_called_once()
    mock_mmcv.imwrite.assert_called_once()


@mock.patch('%s.visualize_utils.mmcv.imshow' % __name__)
@mock.patch('%s.visualize_utils.mmcv.imwrite' % __name__)
def test_imshow_pred_boundary(mock_imshow, mock_imwrite):
    img = './tests/data/test_img1.jpg'
    boundaries_with_scores = [[0, 0, 1, 0, 1, 1, 0, 1, 1]]
    labels = [1]
    file = tempfile.NamedTemporaryFile().name
    visualize_utils.imshow_pred_boundary(
        img, boundaries_with_scores, labels, show=True, out_file=file)
    mock_imwrite.assert_called_once()
    mock_imshow.assert_called_once()


@mock.patch('%s.visualize_utils.mmcv.imshow' % __name__)
@mock.patch('%s.visualize_utils.mmcv.imwrite' % __name__)
def test_imshow_text_char_boundary(mock_imshow, mock_imwrite):

    img = './tests/data/test_img1.jpg'
    text_quads = [[0, 0, 1, 0, 1, 1, 0, 1]]
    boundaries = [[0, 0, 1, 0, 1, 1, 0, 1]]
    char_quads = [[[0, 0, 1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 1, 0, 1]]]
    chars = [['a', 'b']]
    show = True,
    out_file = tempfile.NamedTemporaryFile().name
    visualize_utils.imshow_text_char_boundary(
        img,
        text_quads,
        boundaries,
        char_quads,
        chars,
        show=show,
        out_file=out_file)
    mock_imwrite.assert_called_once()
    mock_imshow.assert_called_once()


@mock.patch('%s.visualize_utils.cv2.drawContours' % __name__)
def test_overlay_mask_img(mock_drawContours):

    img = np.random.rand(10, 10)
    mask = np.zeros((10, 10))
    visualize_utils.overlay_mask_img(img, mask)
    mock_drawContours.assert_called_once()


def test_extract_boundary():
    result = {}

    # test invalid arguments
    with pytest.raises(AssertionError):
        mask_utils.extract_boundary(result)

    result = {'boundary_result': [0, 1]}
    with pytest.raises(AssertionError):
        mask_utils.extract_boundary(result)

    result = {'boundary_result': [[0, 0, 1, 0, 1, 1, 0, 1, 1]]}

    output = mask_utils.extract_boundary(result)
    assert output[2] == [1]
