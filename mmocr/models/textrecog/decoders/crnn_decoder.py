# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import Sequential

from mmocr.models.builder import DECODERS
from mmocr.models.textrecog.layers import BidirectionalLSTM
from .base_decoder import BaseDecoder


@DECODERS.register_module()
class CRNNDecoder(BaseDecoder):

    def __init__(self,
                 in_channels=None,
                 num_classes=None,
                 rnn_flag=False,
                 init_cfg=dict(type='Xavier', layer='Conv2d'),
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.rnn_flag = rnn_flag

        if rnn_flag:
            self.decoder = Sequential(
                BidirectionalLSTM(in_channels, 256, 256),
                BidirectionalLSTM(256, 256, num_classes))
        else:
            self.decoder = nn.Conv2d(
                in_channels, num_classes, kernel_size=1, stride=1)

    def forward_train(self, feat, out_enc, targets_dict, img_metas):
        assert feat.size(2) == 1, 'feature height must be 1'
        if self.rnn_flag:
            x = feat.squeeze(2)  # [N, C, W]
            x = x.permute(2, 0, 1)  # [W, N, C]
            x = self.decoder(x)  # [W, N, C]
            outputs = x.permute(1, 0, 2).contiguous()
        else:
            x = self.decoder(feat)
            x = x.permute(0, 3, 1, 2).contiguous()
            n, w, c, h = x.size()
            outputs = x.view(n, w, c * h)
        return outputs

    def forward_test(self, feat, out_enc, img_metas):
        return self.forward_train(feat, out_enc, None, img_metas)
