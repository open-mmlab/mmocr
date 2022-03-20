# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

import pytest

from mmocr.models.textrecog.convertors import BaseConvertor


def test_base_label_convertor():
    with pytest.raises(NotImplementedError):
        label_convertor = BaseConvertor()
        label_convertor.str2tensor(None)
        label_convertor.tensor2idx(None)

    tmp_dir = tempfile.TemporaryDirectory()
    dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')

    # Test loading a dictionary from file

    # Test the capability of handling different line separator style
    # Set newline='' to preserve the line separators as given in the test file
    # *nix style line separator
    with open(dict_file, 'w', newline='') as fw:
        fw.write('a\nb\n\n \n\n')
    label_convertor = BaseConvertor(dict_file=dict_file)
    assert label_convertor.idx2char == ['a', 'b', ' ']
    # Windows style line separator
    with open(dict_file, 'w', newline='') as fw:
        fw.write('a\r\nb\r\n\r\n \r\n\r\n')
    label_convertor = BaseConvertor(dict_file=dict_file)
    assert label_convertor.idx2char == ['a', 'b', ' ']

    # Ensure it won't parse line separator as a space character
    with open(dict_file, 'w') as fw:
        fw.write('a\nb\n\n\nc\n\n')
    label_convertor = BaseConvertor(dict_file=dict_file)
    assert label_convertor.idx2char == ['a', 'b', 'c']

    # Test loading an illegal dictionary
    # Duplciated characters
    with open(dict_file, 'w') as fw:
        fw.write('a\nb\r\n\n \n\na')
    with pytest.raises(AssertionError):
        label_convertor = BaseConvertor(dict_file=dict_file)

    # Too many characters per line
    with open(dict_file, 'w') as fw:
        fw.write('a\nb\r\nc \n')
    with pytest.raises(
            ValueError,
            match='Expect each line has 0 or 1 character, got 2'
            ' characters at line 3'):
        label_convertor = BaseConvertor(dict_file=dict_file)
    with open(dict_file, 'w') as fw:
        fw.write('   \n')
    with pytest.raises(
            ValueError,
            match='Expect each line has 0 or 1 character, got 3'
            ' characters at line 1'):
        label_convertor = BaseConvertor(dict_file=dict_file)

    # Test creating a dictionary from dict_type
    label_convertor = BaseConvertor(dict_type='DICT36')
    assert len(label_convertor.idx2char) == 36
    with pytest.raises(
            NotImplementedError, match='Dict type DICT100 is not supported'):
        label_convertor = BaseConvertor(dict_type='DICT100')

    # Test creating a dictionary from dict_list
    label_convertor = BaseConvertor(dict_list=['a', 'b', 'c', 'd', ' '])
    assert label_convertor.idx2char == ['a', 'b', 'c', 'd', ' ']

    tmp_dir.cleanup()
