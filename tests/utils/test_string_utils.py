# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmocr.utils import StringStripper


def test_string_strip():
    strip_list = [True, False]
    strip_pos_list = ['both', 'left', 'right']
    strip_str_list = [None, ' ']

    in_str_list = [
        ' hello ', 'hello ', ' hello', ' hello', 'hello ', 'hello ', 'hello',
        'hello', 'hello', 'hello', 'hello', 'hello'
    ]
    out_str_list = [
        'hello', 'hello', 'hello', 'hello', 'hello', 'hello', 'hello', 'hello',
        'hello', 'hello', 'hello', 'hello'
    ]

    for idx1, strip in enumerate(strip_list):
        for idx2, strip_pos in enumerate(strip_pos_list):
            for idx3, strip_str in enumerate(strip_str_list):
                tmp_args = dict(
                    strip=strip, strip_pos=strip_pos, strip_str=strip_str)
                strip_class = StringStripper(**tmp_args)
                i = idx1 * len(strip_pos_list) * len(
                    strip_str_list) + idx2 * len(strip_str_list) + idx3

                assert strip_class(in_str_list[i]) == out_str_list[i]

    with pytest.raises(AssertionError):
        StringStripper(strip='strip')
        StringStripper(strip_pos='head')
        StringStripper(strip_str=['\n', '\t'])
