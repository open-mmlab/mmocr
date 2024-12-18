# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseParser
from .coco_parser import COCOTextDetAnnParser
from .ctw1500_parser import CTW1500AnnParser
from .detext_parser import DetextDetAnnParser
from .funsd_parser import FUNSDTextDetAnnParser
from .icdar_txt_parser import (ICDARTxtTextDetAnnParser,
                               ICDARTxtTextRecogAnnParser)
from .mjsynth_parser import MJSynthAnnParser
from .naf_parser import NAFAnnParser
from .sroie_parser import SROIETextDetAnnParser
from .svt_parser import SVTTextDetAnnParser
from .synthtext_parser import SynthTextAnnParser
from .totaltext_parser import TotaltextTextDetAnnParser
from .wildreceipt_parser import WildreceiptKIEAnnParser

__all__ = [
    'BaseParser', 'ICDARTxtTextDetAnnParser', 'ICDARTxtTextRecogAnnParser',
    'TotaltextTextDetAnnParser', 'WildreceiptKIEAnnParser',
    'COCOTextDetAnnParser', 'SVTTextDetAnnParser', 'FUNSDTextDetAnnParser',
    'SROIETextDetAnnParser', 'NAFAnnParser', 'CTW1500AnnParser',
    'SynthTextAnnParser', 'MJSynthAnnParser', 'DetextDetAnnParser'
]
