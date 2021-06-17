from mmdet.datasets.builder import PIPELINES

from . import PANetTargets


@PIPELINES.register_module()
class PSENetTargets(PANetTargets):
    """Generate the ground truth targets of PSENet: Shape robust text detection
    with progressive scale expansion network.

    [https://arxiv.org/abs/1903.12473]. This code is partially adapted from
    https://github.com/whai362/PSENet.

    Args:
        shrink_ratio(tuple(float)): The ratios for shrinking text instances.
        max_shrink(int): The maximum shrinking distance.
    """

    def __init__(self,
                 shrink_ratio=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4),
                 max_shrink=20):
        super().__init__(shrink_ratio=shrink_ratio, max_shrink=max_shrink)
