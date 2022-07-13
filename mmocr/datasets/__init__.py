# Copyright (c) OpenMMLab. All rights reserved.
from .icdar_dataset import IcdarDataset
from .ocr_dataset import OCRDataset
from .recog_lmdb_dataset import RecogLMDBDataset
from .recog_text_dataset import RecogTextDataset
from .transforms import *  # NOQA
from .wildreceipt_dataset import WildReceiptDataset

__all__ = [
    'IcdarDataset', 'OCRDataset', 'RecogLMDBDataset', 'RecogTextDataset',
    'WildReceiptDataset'
]
