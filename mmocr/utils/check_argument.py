# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


def is_3dlist(x):

    if not isinstance(x, list):
        return False
    if len(x) > 0:
        if isinstance(x[0], list):
            if len(x[0]) > 0:
                return isinstance(x[0][0], list)
            return True
        return False

    return True


def is_2dlist(x):

    if not isinstance(x, list):
        return False
    if len(x) > 0:
        return bool(isinstance(x[0], list))

    return True


def is_ndarray_list(x):

    if not isinstance(x, list):
        return False
    if len(x) > 0:
        return isinstance(x[0], np.ndarray)

    return True


def is_type_list(x, type):

    if not isinstance(x, list):
        return False
    if len(x) > 0:
        return isinstance(x[0], type)

    return True


def is_none_or_type(x, type):

    return isinstance(x, type) or x is None


def equal_len(*argv):
    assert len(argv) > 0

    num_arg = len(argv[0])
    for arg in argv:
        if len(arg) != num_arg:
            return False
    return True


def valid_boundary(x, with_score=True):
    num = len(x)
    if num < 8:
        return False
    if num % 2 == 0 and (not with_score):
        return True
    if num % 2 == 1 and with_score:
        return True

    return False
