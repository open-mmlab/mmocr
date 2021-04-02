import os.path as osp
import tempfile

import numpy as np
import pytest
import torch

from mmocr.models.textrecog.convertors import BaseConvertor, CTCConvertor


def _create_dummy_dict_file(dict_file):
    chars = list('helowrd')
    with open(dict_file, 'w') as fw:
        for char in chars:
            fw.write(char + '\n')


def test_ctc_label_convertor():
    tmp_dir = tempfile.TemporaryDirectory()
    # create dummy data
    dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
    _create_dummy_dict_file(dict_file)

    # test invalid arguments
    with pytest.raises(AssertionError):
        CTCConvertor(5)

    label_convertor = CTCConvertor(dict_file=dict_file, with_unknown=False)
    # test init and parse_chars
    assert label_convertor.num_classes() == 8
    assert len(label_convertor.idx2char) == 8
    assert label_convertor.idx2char[0] == '<BLK>'
    assert label_convertor.char2idx['h'] == 1
    assert label_convertor.unknown_idx is None

    # test encode str to tensor
    strings = ['hell']
    expect_tensor = torch.IntTensor([1, 2, 3, 3])
    targets_dict = label_convertor.str2tensor(strings)
    assert torch.allclose(targets_dict['targets'][0], expect_tensor)
    assert torch.allclose(targets_dict['flatten_targets'], expect_tensor)
    assert torch.allclose(targets_dict['target_lengths'], torch.IntTensor([4]))

    # test decode output to index
    dummy_output = torch.Tensor([[[1, 100, 3, 4, 5, 6, 7, 8],
                                  [100, 2, 3, 4, 5, 6, 7, 8],
                                  [1, 2, 100, 4, 5, 6, 7, 8],
                                  [1, 2, 100, 4, 5, 6, 7, 8],
                                  [100, 2, 3, 4, 5, 6, 7, 8],
                                  [1, 2, 3, 100, 5, 6, 7, 8],
                                  [100, 2, 3, 4, 5, 6, 7, 8],
                                  [1, 2, 3, 100, 5, 6, 7, 8]]])
    indexes, scores = label_convertor.tensor2idx(
        dummy_output, img_metas=[{
            'valid_ratio': 1.0
        }])
    assert np.allclose(indexes, [[1, 2, 3, 3]])

    # test encode_str_label_to_index
    with pytest.raises(AssertionError):
        label_convertor.str2idx('hell')
    tmp_indexes = label_convertor.str2idx(strings)
    assert np.allclose(tmp_indexes, [[1, 2, 3, 3]])

    # test deocde_index_to_str_label
    input_indexes = [[1, 2, 3, 3]]
    with pytest.raises(AssertionError):
        label_convertor.idx2str('hell')
    output_strings = label_convertor.idx2str(input_indexes)
    assert output_strings[0] == 'hell'

    tmp_dir.cleanup()


def test_base_label_convertor():
    with pytest.raises(NotImplementedError):
        label_convertor = BaseConvertor()
        label_convertor.str2tensor(None)
        label_convertor.tensor2idx(None)
