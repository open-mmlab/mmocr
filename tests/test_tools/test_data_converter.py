"""Test orientation check and ignore method."""

import shutil
import tempfile


def test_check_ignore_orientation():
    from tools.data.utils.common \
        import check_ignore_orientation
    img_file = 'tests/data/test_img2.jpg'
    output_file = check_ignore_orientation(img_file)
    assert output_file is img_file

    img_file = 'tests/data/test_img1.jpg'
    tmp_dir = tempfile.TemporaryDirectory()
    dst_file = shutil.copy(img_file, tmp_dir.name)
    output_file = check_ignore_orientation(dst_file)
    assert output_file[-3:] == 'png'
