# Copyright (c) OpenMMLab. All rights reserved.
"""Test orientation check and ignore method."""

import shutil
import tempfile

from mmocr.utils import drop_orientation


def test_drop_orientation():
    img_file = 'tests/data/test_img2.jpg'
    output_file = drop_orientation(img_file)
    assert output_file is img_file

    img_file = 'tests/data/test_img1.jpg'
    tmp_dir = tempfile.TemporaryDirectory()
    dst_file = shutil.copy(img_file, tmp_dir.name)
    output_file = drop_orientation(dst_file)
    assert output_file[-3:] == 'png'
