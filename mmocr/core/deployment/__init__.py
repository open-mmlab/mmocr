# Copyright (c) OpenMMLab. All rights reserved.
from .deploy_utils import (ONNXRuntimeDetector, ONNXRuntimeRecognizer,
                           TensorRTDetector, TensorRTRecognizer)

__all__ = [
    'ONNXRuntimeRecognizer', 'ONNXRuntimeDetector', 'TensorRTDetector',
    'TensorRTRecognizer'
]
