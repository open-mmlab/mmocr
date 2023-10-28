# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseDatasetConfigGenerator
from .textdet_config_generator import TextDetConfigGenerator
from .textrecog_config_generator import TextRecogConfigGenerator
from .textspotting_config_generator import TextSpottingConfigGenerator
from .xfund_config_generator import (XFUNDREConfigGenerator,
                                     XFUNDSERConfigGenerator)

__all__ = [
    'BaseDatasetConfigGenerator', 'TextDetConfigGenerator',
    'TextRecogConfigGenerator', 'TextSpottingConfigGenerator',
    'XFUNDSERConfigGenerator', 'XFUNDREConfigGenerator'
]
