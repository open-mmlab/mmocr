# Copyright (c) OpenMMLab. All rights reserved.
from .nrtr_modality_transformer import NRTRModalityTransform
from .resnet import ResNetOCR
from .resnet31_ocr import ResNet31OCR
from .resnet_abi import ResNetABI
from .shallow_cnn import ShallowCNN
from .very_deep_vgg import VeryDeepVgg

__all__ = [
    'ResNet31OCR', 'VeryDeepVgg', 'NRTRModalityTransform', 'ShallowCNN',
    'ResNetABI', 'ResNetOCR'
]
