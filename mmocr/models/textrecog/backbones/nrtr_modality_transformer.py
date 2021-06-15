import torch.nn as nn
from mmcv.cnn import kaiming_init, uniform_init
from mmdet.models.builder import BACKBONES


@BACKBONES.register_module()
class NRTRModalityTransform(nn.Module):

    def __init__(self, input_channels=3, input_height=32):
        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1)
        self.relu_1 = nn.ReLU(True)
        self.bn_1 = nn.BatchNorm2d(32)

        self.conv_2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1)
        self.relu_2 = nn.ReLU(True)
        self.bn_2 = nn.BatchNorm2d(64)

        feat_height = input_height // 4

        self.linear = nn.Linear(64 * feat_height, 512)

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                uniform_init(m)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.bn_1(x)

        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.bn_2(x)

        n, c, h, w = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(n, w, h * c)
        x = self.linear(x)
        x = x.permute(0, 2, 1).contiguous().view(n, -1, 1, w)
        return x
