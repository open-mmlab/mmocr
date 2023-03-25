# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseDatasetConfigGenerator
from .textdet_config_generator import TextDetConfigGenerator
from .textrecog_config_generator import TextRecogConfigGenerator
from .textspotting_config_generator import TextSpottingConfigGenerator
from .ser_config_generator import SERConfigGenerator

__all__ = [
    'BaseDatasetConfigGenerator', 'TextDetConfigGenerator',
    'TextRecogConfigGenerator', 'TextSpottingConfigGenerator',
    'SERConfigGenerator'
]
