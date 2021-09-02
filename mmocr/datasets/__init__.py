# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets.builder import build_dataloader

from . import utils
from .base_dataset import BaseDataset
from .builder import DATASETS, build_dataset
from .coco import CocoDataset
from .custom import CustomDataset
from .icdar_dataset import IcdarDataset
from .kie_dataset import KIEDataset
from .ner_dataset import NerDataset
from .ocr_dataset import OCRDataset
from .ocr_seg_dataset import OCRSegDataset
from .pipelines import CustomFormatBundle, DBNetTargets, FCENetTargets
from .text_det_dataset import TextDetDataset
from .uniform_concat_dataset import UniformConcatDataset

from .utils import *  # NOQA

__all__ = [
    'DATASETS', 'CocoDataset', 'CustomDataset', 'IcdarDataset',
    'build_dataloader', 'build_dataset', 'BaseDataset', 'OCRDataset',
    'TextDetDataset', 'CustomFormatBundle', 'DBNetTargets', 'OCRSegDataset',
    'KIEDataset', 'FCENetTargets', 'NerDataset', 'UniformConcatDataset'
]

__all__ += utils.__all__
