# Copyright (c) OpenMMLab. All rights reserved.
from .ce_loss import CELoss, SARLoss, TFLoss
from .ctc_loss import CTCLoss
from .seg_loss import SegLoss

__all__ = ['CELoss', 'SARLoss', 'CTCLoss', 'TFLoss', 'SegLoss']
