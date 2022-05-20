# Copyright (c) OpenMMLab. All rights reserved.
from . import detectors, heads, losses, necks, postprocessors
from .detectors import *  # NOQA
from .heads import *  # NOQA
from .losses import *  # NOQA
from .necks import *  # NOQA
from .postprocessors import *  # NOQA

__all__ = (
    heads.__all__ + detectors.__all__ + losses.__all__ + necks.__all__ +
    postprocessors.__all__)
