import torch
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from torch import nn

from mmdet.models.builder import NECKS


class UpBlock(nn.Module):
    """Upsample block for DRRG and TextSnake."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        assert isinstance(in_channels, int)
        assert isinstance(out_channels, int)

        self.conv1x1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1x1(x))
        x = F.relu(self.conv3x3(x))
        x = self.deconv(x)
        return x


@NECKS.register_module()
class FPN_UNet(nn.Module):
    """The class for implementing DRRG and TextSnake U-Net-like FPN.

    DRRG: Deep Relational Reasoning Graph Network for Arbitrary Shape
    Text Detection [https://arxiv.org/abs/2003.07493].
    TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes
    [https://arxiv.org/abs/1807.01544].
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        assert len(in_channels) == 4
        assert isinstance(out_channels, int)

        blocks_out_channels = [out_channels] + [
            min(out_channels * 2**i, 256) for i in range(4)
        ]
        blocks_in_channels = [blocks_out_channels[1]] + [
            in_channels[i] + blocks_out_channels[i + 2] for i in range(3)
        ] + [in_channels[3]]

        self.up4 = nn.ConvTranspose2d(
            blocks_in_channels[4],
            blocks_out_channels[4],
            kernel_size=4,
            stride=2,
            padding=1)
        self.up_block3 = UpBlock(blocks_in_channels[3], blocks_out_channels[3])
        self.up_block2 = UpBlock(blocks_in_channels[2], blocks_out_channels[2])
        self.up_block1 = UpBlock(blocks_in_channels[1], blocks_out_channels[1])
        self.up_block0 = UpBlock(blocks_in_channels[0], blocks_out_channels[0])
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        c2, c3, c4, c5 = x

        x = F.relu(self.up4(c5))

        x = torch.cat([x, c4], dim=1)
        x = F.relu(self.up_block3(x))

        x = torch.cat([x, c3], dim=1)
        x = F.relu(self.up_block2(x))

        x = torch.cat([x, c2], dim=1)
        x = F.relu(self.up_block1(x))

        x = self.up_block0(x)
        # the output should be of the same height and width as backbone input
        return x
