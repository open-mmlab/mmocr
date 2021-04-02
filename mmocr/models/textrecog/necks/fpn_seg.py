import torch.nn.functional as F
from mmcv.runner import auto_fp16

from mmdet.models.builder import NECKS
from mmdet.models.necks import FPN


@NECKS.register_module()
class FPNSeg(FPN):
    """Feature Pyramid Network for segmentation based text recognition.

    Args:
        in_channels (list[int]): Number of input channels for each scale.
        out_channels (int): Number of output channels for each scale.
        num_outs (int): Number of output scales.
        upsample_param (dict | None): Config dict for interpolate layer.
            Default: `dict(scale_factor=1.0, mode='nearest')`
        last_stage_only (bool): If True, output last stage of FPN only.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 upsample_param=None,
                 last_stage_only=True):
        super().__init__(in_channels, out_channels, num_outs)
        self.upsample_param = upsample_param
        self.last_stage_only = last_stage_only

    @auto_fp16()
    def forward(self, inputs):
        outs = super().forward(inputs)

        outs = list(outs)

        if self.upsample_param is not None:
            outs[0] = F.interpolate(outs[0], **self.upsample_param)

        if self.last_stage_only:
            return tuple(outs[0:1])

        return tuple(outs[::-1])
