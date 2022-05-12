# Copyright (c) OpenMMLab. All rights reserved.
from . import common, kie, textdet, textrecog
from .common import *  # NOQA
from .kie import *  # NOQA
from .ner import *  # NOQA
from .textdet import *  # NOQA
from .textrecog import *  # NOQA

__all__ = common.__all__ + kie.__all__ + textdet.__all__ + textrecog.__all__
