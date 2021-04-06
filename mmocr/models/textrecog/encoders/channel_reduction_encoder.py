import torch.nn as nn
from mmcv.cnn import xavier_init

from mmocr.models.builder import ENCODERS
from .base_encoder import BaseEncoder


@ENCODERS.register_module()
class ChannelReductionEncoder(BaseEncoder):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layer = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)

    def forward(self, feat, img_metas=None):
        return self.layer(feat)
