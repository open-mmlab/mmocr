# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmocr.models.builder import DECODERS
from mmocr.models.textrecog.layers import attention_aster_layer
from .base_decoder import BaseDecoder


@DECODERS.register_module()
class ASTERDecoder(BaseDecoder):
    """Decoder for Aster.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        s_Dim (int): The state of dimensions
        Atten_Dim(int):Then attention layer of dimensions:
        max_seq_len:

        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels,
                 num_classes,
                 s_Dim,
                 Atten_Dim,
                 max_seq_len,
                 start_idx=0,
                 padding_idx=92,
                 init_cgf=None,
                 **kwargs):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.s_Dim = s_Dim
        self.Atten_Dim = Atten_Dim
        self.max_seq_len = max_seq_len
        super().__init__(init_cgf=init_cgf)

        self.decoder = attention_aster_layer.Decoder(in_channels, num_classes,
                                                     s_Dim, Atten_Dim)

    def forward_train(self, feat, out_enc, targets_dict, img_metas):
        x = out_enc
        labels = targets_dict['padded_targets'].to(feat.device)
        batch_size = x.size(0)
        state = torch.zeros(1, batch_size, self.s_Dim).to(feat.device)
        outputs = []
        for i in range(self.max_seq_len):
            if i == 0:
                y_pre = torch.zeros(
                    (batch_size)).fill_(self.num_classes).to(feat.device)
            else:
                y_pre = labels[:, i - 1]
            output, state = self.decoder(x, state, y_pre)
            outputs.append(output)
        outputs = torch.cat([_.unsqueeze(1) for _ in outputs], 1)
        return outputs

    def forward_test(self, feat, out_enc, img_metas):
        x = out_enc
        batch_size = x.size(0)
        state = torch.zeros(1, batch_size, self.s_Dim).to(feat.device)
        out_put = []
        for i in range(self.max_seq_len):
            if i == 0:
                y_pre = torch.zeros(
                    (batch_size)).fill_(self.num_classes).to(feat.device)
            else:
                y_pre = predicted
            output, state = self.decoder(x, state, y_pre)
            # out_put.append(output)
            outputs = F.softmax(output, dim=1)
            out_put.append(outputs)
            score, predicted = outputs.max(-1)
        out_put = torch.cat([_.unsqueeze(1) for _ in out_put], 1)
        out_put = out_put[:, 1:, :]
        return out_put
