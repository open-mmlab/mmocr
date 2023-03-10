# Copyright (c) OpenMMLab. All rights reserved.
from .kie_inferencer import KIEInferencer
from .mmocr_inferencer import MMOCRInferencer
from .textdet_inferencer import TextDetInferencer
from .textrec_inferencer import TextRecInferencer
from .textspot_inferencer import TextSpotInferencer

__all__ = [
    'TextDetInferencer', 'TextRecInferencer', 'KIEInferencer',
    'MMOCRInferencer', 'TextSpotInferencer'
]
