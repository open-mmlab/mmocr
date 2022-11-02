# Copyright (c) OpenMMLab. All rights reserved.
from .coco_parser import COCOTextDetAnnParser
from .icdar_txt_parser import (ICDARTxtTextDetAnnParser,
                               ICDARTxtTextRecogAnnParser)
from .totaltext_parser import TotaltextTextDetAnnParser
from .wildreceipt_parser import WildreceiptKIEAnnParser

__all__ = [
    'ICDARTxtTextDetAnnParser', 'ICDARTxtTextRecogAnnParser',
    'TotaltextTextDetAnnParser', 'WildreceiptKIEAnnParser',
    'COCOTextDetAnnParser'
]
