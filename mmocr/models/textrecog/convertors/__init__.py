# Copyright (c) OpenMMLab. All rights reserved.
from .abi import ABIConvertor
from .attn import AttnConvertor
from .base import BaseConvertor
from .ctc import CTCConvertor
<<<<<<< HEAD
=======
from .master import MasterConvertor
>>>>>>> 197de40... fix #794: add MASTER
from .seg import SegConvertor

__all__ = [
    'BaseConvertor', 'CTCConvertor', 'AttnConvertor', 'SegConvertor',
<<<<<<< HEAD
    'ABIConvertor'
=======
    'ABIConvertor', 'MasterConvertor'
>>>>>>> 197de40... fix #794: add MASTER
]
