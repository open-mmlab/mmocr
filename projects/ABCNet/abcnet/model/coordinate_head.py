import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmengine.model import BaseModule
from mmocr.registry import MODELS


@MODELS.register_module()
class CoordinateHead(BaseModule):

    def __init__(self,
                 in_channel=256,
                 conv_num=4,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        mask_convs = list()
        for i in range(conv_num):
            if i == 0:
                mask_conv = ConvModule(
                    in_channels=in_channel + 2,  # 2 for coord
                    out_channels=in_channel,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            else:
                mask_conv = ConvModule(
                    in_channels=in_channel,
                    out_channels=in_channel,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            mask_convs.append(mask_conv)
        self.mask_convs = nn.Sequential(*mask_convs)

    def forward(self, features):
        coord_features = list()
        for feature in features:
            x_range = torch.linspace(
                -1, 1, feature.shape[-1], device=feature.device)
            y_range = torch.linspace(
                -1, 1, feature.shape[-2], device=feature.device)
            y, x = torch.meshgrid(y_range, x_range)
            y = y.expand([feature.shape[0], 1, -1, -1])
            x = x.expand([feature.shape[0], 1, -1, -1])
            coord = torch.cat([x, y], 1)
            feature_with_coord = torch.cat([feature, coord], dim=1)
            feature_with_coord = self.mask_convs(feature_with_coord)
            feature_with_coord = feature_with_coord + feature
            coord_features.append(feature_with_coord)
        return coord_features
