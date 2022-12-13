# Copyright (c) OpenMMLab. All rights reserved.
from .abinet import ABINet
from .base import BaseRecognizer
from .crnn import CRNN
from .encoder_decoder_recognizer import EncoderDecoderRecognizer
from .master import MASTER
from .nrtr import NRTR
from .robust_scanner import RobustScanner
from .sar import SARNet
from .satrn import SATRN
from .svtr import SVTR

__all__ = [
    'BaseRecognizer', 'EncoderDecoderRecognizer', 'CRNN', 'SARNet', 'NRTR',
    'RobustScanner', 'SATRN', 'ABINet', 'MASTER', 'SVTR'
]
