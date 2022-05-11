# Copyright (c) OpenMMLab. All rights reserved.
from . import utils
from .base_dataset import BaseDataset
from .builder import LOADERS, PARSERS, TRANSFORMS
from .icdar_dataset import IcdarDataset
# TODO: check this
from .kie_dataset import KIEDataset
from .ner_dataset import NerDataset
from .ocr_dataset import OCRDataset
from .ocr_seg_dataset import OCRSegDataset
from .openset_kie_dataset import OpensetKIEDataset
from .pipelines import DBNetTargets, FCENetTargets
from .text_det_dataset import TextDetDataset
from .uniform_concat_dataset import UniformConcatDataset
from .utils import *  # NOQA

__all__ = [
    'IcdarDataset', 'BaseDataset', 'OCRDataset', 'TextDetDataset',
    'KIEDataset', 'DBNetTargets', 'OCRSegDataset', 'FCENetTargets',
    'NerDataset', 'UniformConcatDataset', 'OpensetKIEDataset', 'TRANSFORMS',
    'PARSERS', 'LOADERS'
]

__all__ += utils.__all__
