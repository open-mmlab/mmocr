# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

import mmocr.utils as utils


def test_is_3dlist():

    assert utils.is_3dlist([])
    assert utils.is_3dlist([[]])
    assert utils.is_3dlist([[[]]])
    assert utils.is_3dlist([[[1]]])
    assert not utils.is_3dlist([[1, 2]])
    assert not utils.is_3dlist([[np.array([1, 2])]])


def test_is_2dlist():

    assert utils.is_2dlist([])
    assert utils.is_2dlist([[]])
    assert utils.is_2dlist([[1]])


def test_is_type_list():
    assert utils.is_type_list([], int)
    assert utils.is_type_list([], float)
    assert utils.is_type_list([np.array([])], np.ndarray)
    assert utils.is_type_list([1], int)
    assert utils.is_type_list(['str'], str)


def test_is_none_or_type():

    assert utils.is_none_or_type(None, int)
    assert utils.is_none_or_type(1.0, float)
    assert utils.is_none_or_type(np.ndarray([]), np.ndarray)
    assert utils.is_none_or_type(1, int)
    assert utils.is_none_or_type('str', str)


def test_valid_boundary():

    x = [0, 0, 1, 0, 1, 1, 0, 1]
    assert not utils.valid_boundary(x, True)
    assert not utils.valid_boundary([0])
    assert utils.valid_boundary(x, False)
    x = [0, 0, 1, 0, 1, 1, 0, 1, 1]
    assert utils.valid_boundary(x, True)
