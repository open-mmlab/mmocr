# Copyright (c) OpenMMLab. All rights reserved.
from .nrtr_modality_transformer import NRTRModalityTransform
from .resnet31_ocr import ResNet31OCR
from .shallow_cnn import ShallowCNN
from .very_deep_vgg import VeryDeepVgg

__all__ = ['ResNet31OCR', 'VeryDeepVgg', 'NRTRModalityTransform', 'ShallowCNN']
