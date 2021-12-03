# Copyright (c) OpenMMLab. All rights reserved.
from .base_text_detector import BaseTextDetector
from .dbnet import DBNet
from .drrg import DRRG
from .fcenet import FCENet
from .ocr_mask_rcnn import OCRMaskRCNN
from .panet import PANet
from .psenet import PSENet
from .single_stage_text_detector import SingleStageTextDetector
from .textsnake import TextSnake

__all__ = [
    'BaseTextDetector', 'SingleStageTextDetector', 'OCRMaskRCNN', 'DBNet',
    'PANet', 'PSENet', 'TextSnake', 'FCENet', 'DRRG'
]
