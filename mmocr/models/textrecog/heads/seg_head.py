import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models.builder import HEADS
from torch import nn


@HEADS.register_module()
class SegHead(nn.Module):
    """Head for segmentation based text recognition.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        upsample_param (dict | None): Config dict for interpolation layer.
            Default: `dict(scale_factor=1.0, mode='nearest')`
    """

    def __init__(self, in_channels=128, num_classes=37, upsample_param=None):
        super().__init__()
        assert isinstance(num_classes, int)
        assert num_classes > 0
        assert upsample_param is None or isinstance(upsample_param, dict)

        self.upsample_param = upsample_param

        self.seg_conv = ConvModule(
            in_channels,
            in_channels,
            3,
            stride=1,
            padding=1,
            norm_cfg=dict(type='BN'))

        # prediction
        self.pred_conv = nn.Conv2d(
            in_channels, num_classes, kernel_size=1, stride=1, padding=0)

    def init_weights(self):
        pass

    def forward(self, out_neck):

        seg_map = self.seg_conv(out_neck[-1])
        seg_map = self.pred_conv(seg_map)

        if self.upsample_param is not None:
            seg_map = F.interpolate(seg_map, **self.upsample_param)

        return seg_map
