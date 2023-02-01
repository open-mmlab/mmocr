# Copyright (c) OpenMMLab. All rights reserved.
from .adel_dataset import AdelDataset
from .transforms.spts_transforms import (Bezier2Polygon, ConvertText,
                                         LoadOCRAnnotationsWithBezier,
                                         Polygon2Bezier, RescaleToShortSide)

__all__ = [
    'AdelDataset', 'LoadOCRAnnotationsWithBezier', 'Bezier2Polygon',
    'Polygon2Bezier', 'ConvertText', 'RescaleToShortSide'
]
