import mmcv
import mmdet

from .version import __version__, short_version


def digit_version(version_str):
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return digit_version


mmcv_minimum_version = '1.3.8'
mmcv_maximum_version = '1.4.0'
mmcv_version = digit_version(mmcv.__version__)

assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version <= digit_version(mmcv_maximum_version)), \
    f'MMCV {mmcv.__version__} is incompatible with MMOCR {__version__}. ' \
    f'Please use MMCV >= {mmcv_minimum_version}, ' \
    f'<= {mmcv_maximum_version} instead.'

mmdet_minimum_version = '2.14.0'
mmdet_maximum_version = '2.20.0'
mmdet_version = digit_version(mmdet.__version__)

assert (mmdet_version >= digit_version(mmdet_minimum_version)
        and mmdet_version <= digit_version(mmdet_maximum_version)), \
    f'MMDetection {mmdet.__version__} is incompatible ' \
    f'with MMOCR {__version__}. ' \
    f'Please use MMDetection >= {mmdet_minimum_version}, ' \
    f'<= {mmdet_maximum_version} instead.'

__all__ = ['__version__', 'short_version']
