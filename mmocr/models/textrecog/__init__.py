# Copyright (c) OpenMMLab. All rights reserved.
from . import (backbones, convertors, decoders, dictionary, encoders, fusers,
               heads, losses, necks, plugins, postprocessors, preprocessors,
               recognizers)
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
from .postprocessors import *  # NOQA
from .preprocessors import *  # NOQA
from .recognizers import *  # NOQA

__all__ = (
    backbones.__all__ + convertors.__all__ + decoders.__all__ +
    encoders.__all__ + heads.__all__ + losses.__all__ + necks.__all__ +
    preprocessors.__all__ + recognizers.__all__ + fusers.__all__ +
    plugins.__all__ + dictionary.__all__ + postprocessors.__all__)
