from .base_textdet_targets import BaseTextDetTargets
from .dbnet_targets import DBNetTargets
from .drrg_targets import DRRGTargets
from .fcenet_targets import FCENetTargets
from .panet_targets import PANetTargets
from .psenet_targets import PSENetTargets
from .textsnake_targets import TextSnakeTargets

__all__ = [
    'BaseTextDetTargets', 'PANetTargets', 'PSENetTargets', 'DBNetTargets',
    'FCENetTargets', 'TextSnakeTargets', 'DRRGTargets'
]
