# Copyright (c) OpenMMLab. All rights reserved.
from .attn_postprocessor import AttentionPostprocessor
from .base_textrecog_postprocessor import BaseTextRecogPostprocessor
from .ctc_postprocessor import CTCPostProcessor

__all__ = [
    'BaseTextRecogPostprocessor', 'AttentionPostprocessor', 'CTCPostProcessor'
]
