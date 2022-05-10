# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

import numpy as np
import pytest

from mmocr.datasets.pipelines.ocr_seg_targets import OCRSegTargets


def _create_dummy_dict_file(dict_file):
    chars = list('0123456789')
    with open(dict_file, 'w') as fw:
        for char in chars:
            fw.write(char + '\n')


def test_ocr_segm_targets():
    tmp_dir = tempfile.TemporaryDirectory()
    # create dummy dict file
    dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
    _create_dummy_dict_file(dict_file)
    # dummy label convertor
    label_convertor = dict(
        type='SegConvertor',
        dict_file=dict_file,
        with_unknown=True,
        lower=True)
    # test init
    with pytest.raises(AssertionError):
        OCRSegTargets(None, 0.5, 0.5)
    with pytest.raises(AssertionError):
        OCRSegTargets(label_convertor, '1by2', 0.5)
    with pytest.raises(AssertionError):
        OCRSegTargets(label_convertor, 0.5, 2)

    ocr_seg_tgt = OCRSegTargets(label_convertor, 0.5, 0.5)
    # test generate kernels
    img_size = (8, 8)
    pad_size = (8, 10)
    char_boxes = [[2, 2, 6, 6]]
    char_idxs = [2]

    with pytest.raises(AssertionError):
        ocr_seg_tgt.generate_kernels(8, pad_size, char_boxes, char_idxs, 0.5,
                                     True)
    with pytest.raises(AssertionError):
        ocr_seg_tgt.generate_kernels(img_size, pad_size, [2, 2, 6, 6],
                                     char_idxs, 0.5, True)
    with pytest.raises(AssertionError):
        ocr_seg_tgt.generate_kernels(img_size, pad_size, char_boxes, 2, 0.5,
                                     True)

    attn_tgt = ocr_seg_tgt.generate_kernels(
        img_size, pad_size, char_boxes, char_idxs, 0.5, binary=True)
    expect_attn_tgt = [[0, 0, 0, 0, 0, 0, 0, 0, 255, 255],
                       [0, 0, 0, 0, 0, 0, 0, 0, 255, 255],
                       [0, 0, 0, 0, 0, 0, 0, 0, 255, 255],
                       [0, 0, 0, 1, 1, 1, 0, 0, 255, 255],
                       [0, 0, 0, 1, 1, 1, 0, 0, 255, 255],
                       [0, 0, 0, 1, 1, 1, 0, 0, 255, 255],
                       [0, 0, 0, 0, 0, 0, 0, 0, 255, 255],
                       [0, 0, 0, 0, 0, 0, 0, 0, 255, 255]]
    assert np.allclose(attn_tgt, np.array(expect_attn_tgt, dtype=np.int32))

    segm_tgt = ocr_seg_tgt.generate_kernels(
        img_size, pad_size, char_boxes, char_idxs, 0.5, binary=False)
    expect_segm_tgt = [[0, 0, 0, 0, 0, 0, 0, 0, 255, 255],
                       [0, 0, 0, 0, 0, 0, 0, 0, 255, 255],
                       [0, 0, 0, 0, 0, 0, 0, 0, 255, 255],
                       [0, 0, 0, 2, 2, 2, 0, 0, 255, 255],
                       [0, 0, 0, 2, 2, 2, 0, 0, 255, 255],
                       [0, 0, 0, 2, 2, 2, 0, 0, 255, 255],
                       [0, 0, 0, 0, 0, 0, 0, 0, 255, 255],
                       [0, 0, 0, 0, 0, 0, 0, 0, 255, 255]]
    assert np.allclose(segm_tgt, np.array(expect_segm_tgt, dtype=np.int32))

    # test __call__
    results = {}
    results['img_shape'] = (4, 4, 3)
    results['resize_shape'] = (8, 8, 3)
    results['pad_shape'] = (8, 10)
    results['ann_info'] = {}
    results['ann_info']['char_rects'] = [[1, 1, 3, 3]]
    results['ann_info']['chars'] = ['1']

    results = ocr_seg_tgt(results)
    assert results['mask_fields'] == ['gt_kernels']
    assert np.allclose(results['gt_kernels'].masks[0],
                       np.array(expect_attn_tgt, dtype=np.int32))
    assert np.allclose(results['gt_kernels'].masks[1],
                       np.array(expect_segm_tgt, dtype=np.int32))

    tmp_dir.cleanup()
