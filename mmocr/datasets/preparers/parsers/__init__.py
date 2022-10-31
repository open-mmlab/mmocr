# Copyright (c) OpenMMLab. All rights reserved.
from .coco_parser import COCOTextDetAnnParser
from .ic15_parser import ICDAR2015TextDetAnnParser, ICDAR2015TextRecogAnnParser
from .totaltext_parser import TotaltextTextDetAnnParser
from .wildreceipt_parser import WildreceiptKIEAnnParser

__all__ = [
    'ICDAR2015TextDetAnnParser', 'ICDAR2015TextRecogAnnParser',
    'TotaltextTextDetAnnParser', 'WildreceiptKIEAnnParser',
    'COCOTextDetAnnParser'
]
