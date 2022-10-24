# Copyright (c) OpenMMLab. All rights reserved.
from .ic15_parser import ICDAR2015TextDetParser, ICDAR2015TextRecogParser
from .totaltext_parser import TotaltextTextDetParser
from .wildreceipt import WildreceiptKIEParser

__all__ = [
    'ICDAR2015TextDetParser', 'ICDAR2015TextRecogParser',
    'TotaltextTextDetParser', 'WildreceiptKIEParser'
]
