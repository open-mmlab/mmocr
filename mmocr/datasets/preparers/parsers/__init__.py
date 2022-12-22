# Copyright (c) OpenMMLab. All rights reserved.
from .coco_parser import COCOTextDetAnnParser
from .icdar_txt_parser import (ICDARTxtTextDetAnnParser,
                               ICDARTxtTextRecogAnnParser)
from .svt_parser import SVTTextDetAnnParser
from .totaltext_parser import TotaltextTextDetAnnParser
from .wildreceipt_parser import WildreceiptKIEAnnParser
from .sroie_parser import SROIETextDetAnnParser

__all__ = [
    'ICDARTxtTextDetAnnParser', 'ICDARTxtTextRecogAnnParser',
    'TotaltextTextDetAnnParser', 'WildreceiptKIEAnnParser',
    'COCOTextDetAnnParser', 'SVTTextDetAnnParser', 'SROIETextDetAnnParser'
]
