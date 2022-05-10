# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

import numpy as np
import pytest
import torch

from mmocr.models.textrecog.convertors import ABIConvertor, AttnConvertor


def _create_dummy_dict_file(dict_file):
    characters = list('helowrd')
    with open(dict_file, 'w') as fw:
        for char in characters:
            fw.write(char + '\n')


def test_attn_label_convertor():
    tmp_dir = tempfile.TemporaryDirectory()
    # create dummy data
    dict_file = osp.join(tmp_dir.name, 'fake_dict.txt')
    _create_dummy_dict_file(dict_file)

    # test invalid arguments
    with pytest.raises(NotImplementedError):
        AttnConvertor(5)
    with pytest.raises(AssertionError):
        AttnConvertor('DICT90', dict_file, '1')
    with pytest.raises(AssertionError):
        AttnConvertor('DICT90', dict_file, True, '1')

    label_convertor = AttnConvertor(dict_file=dict_file, max_seq_len=10)
    # test init and parse_dict
    assert label_convertor.num_classes() == 10
    assert len(label_convertor.idx2char) == 10
    assert label_convertor.idx2char[0] == 'h'
    assert label_convertor.idx2char[1] == 'e'
    assert label_convertor.idx2char[-3] == '<UKN>'
    assert label_convertor.char2idx['h'] == 0
    assert label_convertor.unknown_idx == 7

    # test encode str to tensor
    strings = ['hell']
    targets_dict = label_convertor.str2tensor(strings)
    assert torch.allclose(targets_dict['targets'][0],
                          torch.LongTensor([0, 1, 2, 2]))
    assert torch.allclose(targets_dict['padded_targets'][0],
                          torch.LongTensor([8, 0, 1, 2, 2, 8, 9, 9, 9, 9]))

    # test decode output to index
    dummy_output = torch.Tensor([[[100, 2, 3, 4, 5, 6, 7, 8, 9],
                                  [1, 100, 3, 4, 5, 6, 7, 8, 9],
                                  [1, 2, 100, 4, 5, 6, 7, 8, 9],
                                  [1, 2, 100, 4, 5, 6, 7, 8, 9],
                                  [1, 2, 3, 4, 5, 6, 7, 8, 100],
                                  [1, 2, 3, 4, 5, 6, 7, 100, 9],
                                  [1, 2, 3, 4, 5, 6, 7, 100, 9],
                                  [1, 2, 3, 4, 5, 6, 7, 100, 9],
                                  [1, 2, 3, 4, 5, 6, 7, 100, 9],
                                  [1, 2, 3, 4, 5, 6, 7, 100, 9]]])
    indexes, scores = label_convertor.tensor2idx(dummy_output)
    assert np.allclose(indexes, [[0, 1, 2, 2]])

    # test encode_str_label_to_index
    with pytest.raises(AssertionError):
        label_convertor.str2idx('hell')
    tmp_indexes = label_convertor.str2idx(strings)
    assert np.allclose(tmp_indexes, [[0, 1, 2, 2]])

    # test decode_index to str_label
    input_indexes = [[0, 1, 2, 2]]
    with pytest.raises(AssertionError):
        label_convertor.idx2str('hell')
    output_strings = label_convertor.idx2str(input_indexes)
    assert output_strings[0] == 'hell'

    tmp_dir.cleanup()


def test_abi_label_convertor():
    tmp_dir = tempfile.TemporaryDirectory()
    # create dummy data
    dict_file = osp.join(tmp_dir.name, 'fake_dict.txt')
    _create_dummy_dict_file(dict_file)

    label_convertor = ABIConvertor(dict_file=dict_file, max_seq_len=10)

    label_convertor.end_idx
    # test encode str to tensor
    strings = ['hell']
    targets_dict = label_convertor.str2tensor(strings)
    assert torch.allclose(targets_dict['targets'][0],
                          torch.LongTensor([0, 1, 2, 2, 8]))
    assert torch.allclose(targets_dict['padded_targets'][0],
                          torch.LongTensor([8, 0, 1, 2, 2, 8, 9, 9, 9, 9]))

    strings = ['hellhellhell']
    targets_dict = label_convertor.str2tensor(strings)
    assert torch.allclose(targets_dict['targets'][0],
                          torch.LongTensor([0, 1, 2, 2, 0, 1, 2, 2, 0, 8]))
    assert torch.allclose(targets_dict['padded_targets'][0],
                          torch.LongTensor([8, 0, 1, 2, 2, 0, 1, 2, 2, 0]))

    tmp_dir.cleanup()
