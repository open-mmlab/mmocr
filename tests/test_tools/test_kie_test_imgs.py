import tempfile
from glob import glob

from mmcv.utils import check_python_script


def test_kie_test_imgs():
    config = 'tests/data/configs/sdmgr_novisual_1e_toy.py'
    ckpt = ('https://download.openmmlab.com/mmocr/kie/sdmgr/'
            'sdmgr_novisual_60e_wildreceipt_20210517-a44850da.pth')

    with tempfile.TemporaryDirectory() as tmpdir:
        check_python_script(
            f'./tools/kie_test_imgs.py {config} {ckpt} --show-dir {tmpdir}')
        assert len(glob(f'{tmpdir}/*/*/*/*.jpeg')) == 2
