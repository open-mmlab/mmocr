# Copyright (c) OpenMMLab. All rights reserved.
"""Collecting some commonly used type hint in MMOCR."""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from mmengine.config import ConfigDict
from mmengine.structures import InstanceData, LabelData
from mmocr import digit_version
from mmocr.structures import (KIEDataSample, TextDetDataSample,
                              TextRecogDataSample)

# Config
ConfigType = Union[ConfigDict, Dict]
OptConfigType = Optional[ConfigType]
MultiConfig = Union[ConfigType, List[ConfigType]]
OptMultiConfig = Optional[MultiConfig]
InitConfigType = Union[Dict, List[Dict]]
OptInitConfigType = Optional[InitConfigType]

# Data
InstanceList = List[InstanceData]
OptInstanceList = Optional[InstanceList]
LabelList = List[LabelData]
OptLabelList = Optional[LabelList]
RecSampleList = List[TextRecogDataSample]
DetSampleList = List[TextDetDataSample]
KIESampleList = List[KIEDataSample]
OptRecSampleList = Optional[RecSampleList]
OptDetSampleList = Optional[DetSampleList]
OptKIESampleList = Optional[KIESampleList]

OptTensor = Optional[torch.Tensor]

RecForwardResults = Union[Dict[str, torch.Tensor], List[TextRecogDataSample],
                          Tuple[torch.Tensor], torch.Tensor]

# Visualization
ColorType = Union[str, Tuple, List[str], List[Tuple]]

ArrayLike = 'ArrayLike'
if digit_version(np.__version__) >= digit_version('1.20.0'):
    from numpy.typing import ArrayLike as NP_ARRAY_LIKE
    ArrayLike = NP_ARRAY_LIKE

RangeType = Sequence[Tuple[int, int]]
