import torch.nn as nn

from mmocr.models.builder import STAGES


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def conv1x1(in_planes, out_planes):
    """1x1 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False)


@STAGES.register_module()
class BasicStage(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 use_conv1x1=True,
                 stride=1,
                 downsample=None):
        super(BasicStage, self).__init__()
        if use_conv1x1:
            self.conv1 = conv1x1(inplanes, planes)
            self.conv2 = conv3x3(planes, planes, stride)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


@STAGES.register_module()
class Stage_31(BasicStage):

    def __init__(self,
                 inplanes,
                 planes,
                 use_conv1x1=True,
                 stride=1,
                 downsample=None,
                 pool_cfg=None):
        super().__init__(inplanes, planes, use_conv1x1, stride,
                downsample).__init__()
        self.use_maxpool = False
        if pool_cfg:
            self.maxpooling = nn.MaxPool2d(**pool_cfg)
            self.use_maxpool = True
        self.conv3 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x):
        if self.use_maxpool:
            x = self.maxpooling(x)
            
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        out = self.conv3(out)
        return out


@STAGES.register_module()
class stage_master(BasicStage):
    pass
