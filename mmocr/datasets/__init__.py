# Copyright (c) OpenMMLab. All rights reserved.
from .builder import LOADERS, PARSERS
from .icdar_dataset import IcdarDataset
from .ocr_dataset import OCRDataset
from .ocr_seg_dataset import OCRSegDataset
from .pipelines import *  # NOQA
from .recog_lmdb_dataset import RecogLMDBDataset
from .recog_text_dataset import RecogTextDataset

__all__ = [
    'IcdarDataset', 'OCRDataset', 'OCRSegDataset', 'PARSERS', 'LOADERS',
    'RecogLMDBDataset', 'RecogTextDataset'
]
