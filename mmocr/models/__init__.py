from mmdet.models.builder import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS,
                                  build_backbone, build_detector, build_loss)
from . import common, kie, textdet, textrecog
from .builder import (CONVERTORS, DECODERS, ENCODERS, PREPROCESSOR,
                      build_convertor, build_decoder, build_encoder,
                      build_preprocessor)

from .common import *  # NOQA
from .kie import *  # NOQA
from .textdet import *  # NOQA
from .textrecog import *  # NOQA

__all__ = [
    'BACKBONES', 'DETECTORS', 'HEADS', 'LOSSES', 'NECKS', 'build_backbone',
    'build_detector', 'build_loss', 'CONVERTORS', 'ENCODERS', 'DECODERS',
    'PREPROCESSOR', 'build_convertor', 'build_encoder', 'build_decoder',
    'build_preprocessor'
]
__all__ += common.__all__ + kie.__all__ + textdet.__all__ + textrecog.__all__
