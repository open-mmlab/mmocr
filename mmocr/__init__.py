# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import mmdet
from packaging.version import parse

from .version import __version__, short_version


def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions. For pre-release
    versions: alpha < beta < rc.
    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Default: 4.
    Returns:
        tuple[int]: The version info in digits (integers).
    """
    version = parse(version_str)
    assert version.release, f'failed to parse version {version_str}'
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {'a': -3, 'b': -2, 'rc': -1}
        val = -4
        # version.pre can be None
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(f'unknown prerelease version {version.pre[0]}, '
                              'version checking may go wrong')
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])
    else:
        release.extend([0, 0])
    return tuple(release)


mmcv_minimum_version = '2.0.0rc1'
mmcv_maximum_version = '2.1.0'
mmcv_version = digit_version(mmcv.__version__)

assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version < digit_version(mmcv_maximum_version)), \
    f'MMCV {mmcv.__version__} is incompatible with MMOCR {__version__}. ' \
    f'Please use MMCV >= {mmcv_minimum_version}, ' \
    f'< {mmcv_maximum_version} instead.'

mmdet_minimum_version = '3.0.0rc0'
mmdet_maximum_version = '3.1.0'
mmdet_version = digit_version(mmdet.__version__)

assert (mmdet_version >= digit_version(mmdet_minimum_version)
        and mmdet_version < digit_version(mmdet_maximum_version)), \
    f'MMDetection {mmdet.__version__} is incompatible ' \
    f'with MMOCR {__version__}. ' \
    f'Please use MMDetection >= {mmdet_minimum_version}, ' \
    f'< {mmdet_maximum_version} instead.'

__all__ = ['__version__', 'short_version', 'digit_version']
