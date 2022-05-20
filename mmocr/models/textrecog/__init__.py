# Copyright (c) OpenMMLab. All rights reserved.
from . import (backbones, convertors, decoders, dictionary, encoders, fusers,
               heads, losses, necks, plugins, postprocessor, preprocessor,
               recognizer)
from .backbones import *  # NOQA
from .convertors import *  # NOQA
from .decoders import *  # NOQA
from .dictionary import *  # NOQA
from .encoders import *  # NOQA
from .fusers import *  # NOQA
from .heads import *  # NOQA
from .losses import *  # NOQA
from .necks import *  # NOQA
from .plugins import *  # NOQA
from .postprocessor import *  # NOQA
from .preprocessor import *  # NOQA
from .recognizer import *  # NOQA

__all__ = (
    backbones.__all__ + convertors.__all__ + decoders.__all__ +
    encoders.__all__ + heads.__all__ + losses.__all__ + necks.__all__ +
    preprocessor.__all__ + recognizer.__all__ + fusers.__all__ +
    plugins.__all__ + dictionary.__all__ + postprocessor.__all__)
