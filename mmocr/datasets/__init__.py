# Copyright (c) OpenMMLab. All rights reserved.
from .builder import LOADERS, PARSERS
from .icdar_dataset import IcdarDataset
from .ocr_dataset import OCRDataset
from .ocr_seg_dataset import OCRSegDataset
from .pipelines import *  # NOQA
from .utils import *  # NOQA

__all__ = ['IcdarDataset', 'OCRDataset', 'OCRSegDataset', 'PARSERS', 'LOADERS']
