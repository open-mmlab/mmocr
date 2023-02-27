# Copyright (c) OpenMMLab. All rights reserved.

import mmcv
import mmdet

try:
    import mmengine
    from mmengine.utils import digit_version
except ImportError:
    mmengine = None
    from mmcv import digit_version

from .version import __version__, short_version

mmcv_minimum_version = '2.0.0rc4'
mmcv_maximum_version = '2.1.0'
mmcv_version = digit_version(mmcv.__version__)
if mmengine is not None:
    mmengine_minimum_version = '0.6.0'
    mmengine_maximum_version = '1.0.0'
    mmengine_version = digit_version(mmengine.__version__)

if mmcv_version < digit_version('2.0.0rc0') or mmdet.__version__ < '3.0.0rc0':
    raise RuntimeError(
        'MMOCR 1.0 only runs with MMEngine, MMCV 2.0.0rc0+ and '
        'MMDetection 3.0.0rc0+, but got MMCV '
        f'{mmcv.__version__} and MMDetection '
        f'{mmdet.__version__}. For more information, please refer to '
        'https://mmocr.readthedocs.io/en/dev-1.x/migration/overview.html'
    )  # noqa

assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version < digit_version(mmcv_maximum_version)), \
    f'MMCV {mmcv.__version__} is incompatible with MMOCR {__version__}. ' \
    f'Please use MMCV >= {mmcv_minimum_version}, ' \
    f'< {mmcv_maximum_version} instead.'

assert (mmengine_version >= digit_version(mmengine_minimum_version)
        and mmengine_version < digit_version(mmengine_maximum_version)), \
    f'MMEngine=={mmengine.__version__} is used but incompatible. ' \
    f'Please install mmengine>={mmengine_minimum_version}, ' \
    f'<{mmengine_maximum_version}.'

mmdet_minimum_version = '3.0.0rc5'
mmdet_maximum_version = '3.1.0'
mmdet_version = digit_version(mmdet.__version__)

assert (mmdet_version >= digit_version(mmdet_minimum_version)
        and mmdet_version < digit_version(mmdet_maximum_version)), \
    f'MMDetection {mmdet.__version__} is incompatible ' \
    f'with MMOCR {__version__}. ' \
    f'Please use MMDetection >= {mmdet_minimum_version}, ' \
    f'< {mmdet_maximum_version} instead.'

__all__ = ['__version__', 'short_version', 'digit_version']
