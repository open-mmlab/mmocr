# Copyright (c) OpenMMLab. All rights reserved.


def is_3dlist(x):
    """check x is 3d-list([[[1], []]]) or 2d empty list([[], []]) or 1d empty
    list([])"""
    if not isinstance(x, list):
        return False
    if len(x) == 0:
        return True
    for sub_x in x:
        if not is_2dlist(sub_x):
            return False

    return True


def is_2dlist(x):
    """check x is 2d-list([[1], []]) or 1d empty list([])."""
    if not isinstance(x, list):
        return False
    if len(x) == 0:
        return True

    return all(isinstance(item, list) for item in x)


def is_type_list(x, type):

    if not isinstance(x, list):
        return False

    return all(isinstance(item, type) for item in x)


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
