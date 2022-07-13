# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from mmocr.utils import is_on_same_line, sort_points, stitch_boxes_into_lines


def test_box_on_line():
    # regular boxes
    box1 = [0, 0, 1, 0, 1, 1, 0, 1]
    box2 = [2, 0.5, 3, 0.5, 3, 1.5, 2, 1.5]
    box3 = [4, 0.8, 5, 0.8, 5, 1.8, 4, 1.8]
    assert is_on_same_line(box1, box2, 0.5)
    assert not is_on_same_line(box1, box3, 0.5)

    # irregular box4
    box4 = [0, 0, 1, 1, 1, 2, 0, 1]
    box5 = [2, 1.5, 3, 1.5, 3, 2.5, 2, 2.5]
    box6 = [2, 1.6, 3, 1.6, 3, 2.6, 2, 2.6]
    assert is_on_same_line(box4, box5, 0.5)
    assert not is_on_same_line(box4, box6, 0.5)


def test_stitch_boxes_into_lines():
    boxes = [  # regular boxes
        [0, 0, 1, 0, 1, 1, 0, 1],
        [2, 0.5, 3, 0.5, 3, 1.5, 2, 1.5],
        [3, 1.2, 4, 1.2, 4, 2.2, 3, 2.2],
        [5, 0.5, 6, 0.5, 6, 1.5, 5, 1.5],
        # irregular box
        [6, 1.5, 7, 1.25, 7, 1.75, 6, 1.75]
    ]
    raw_input = [{'box': boxes[i], 'text': str(i)} for i in range(len(boxes))]
    result = stitch_boxes_into_lines(raw_input, 1, 0.5)
    # Final lines: [0, 1], [2], [3, 4]
    # box 0, 1, 3, 4 are on the same line but box 3 is 2 pixels away from box 1
    # box 3 and 4 are on the same line since the length of overlapping part >=
    # 0.5 * the y-axis length of box 5
    expected_result = [{
        'box': [0, 0, 3, 0, 3, 1.5, 0, 1.5],
        'text': '0 1'
    }, {
        'box': [3, 1.2, 4, 1.2, 4, 2.2, 3, 2.2],
        'text': '2'
    }, {
        'box': [5, 0.5, 7, 0.5, 7, 1.75, 5, 1.75],
        'text': '3 4'
    }]
    result.sort(key=lambda x: x['box'][0])
    expected_result.sort(key=lambda x: x['box'][0])
    assert result == expected_result


def test_sort_points():
    points = np.array([[1, 1], [0, 0], [1, -1], [2, -2], [0, 2], [1, 1],
                       [0, 1], [-1, 1], [-1, -1]])
    target = np.array([[-1, -1], [0, 0], [-1, 1], [0, 1], [0, 2], [1, 1],
                       [1, 1], [2, -2], [1, -1]])
    assert np.allclose(target, sort_points(points))

    points = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    target = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    assert np.allclose(target, sort_points(points))

    points = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    assert np.allclose(target, sort_points(points))

    with pytest.raises(AssertionError):
        sort_points([1, 2])
