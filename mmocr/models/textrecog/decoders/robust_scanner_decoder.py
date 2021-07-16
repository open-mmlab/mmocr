import torch
import torch.nn as nn
import torch.nn.functional as F

from mmocr.models.builder import DECODERS, build_decoder
from mmocr.models.textrecog.layers import RobustScannerFusionLayer
from .base_decoder import BaseDecoder


@DECODERS.register_module()
class RobustScannerDecoder(BaseDecoder):

    def __init__(self,
                 num_classes=None,
                 dim_input=512,
                 dim_model=128,
                 max_seq_len=40,
                 start_idx=0,
                 mask=True,
                 padding_idx=None,
                 encode_value=False,
                 hybrid_decoder=None,
                 position_decoder=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.dim_input = dim_input
        self.dim_model = dim_model
        self.max_seq_len = max_seq_len
        self.encode_value = encode_value
        self.start_idx = start_idx
        self.padding_idx = padding_idx
        self.mask = mask

        # init hybrid decoder
        hybrid_decoder.update(num_classes=self.num_classes)
        hybrid_decoder.update(dim_input=self.dim_input)
        hybrid_decoder.update(dim_model=self.dim_model)
        hybrid_decoder.update(start_idx=self.start_idx)
        hybrid_decoder.update(padding_idx=self.padding_idx)
        hybrid_decoder.update(max_seq_len=self.max_seq_len)
        hybrid_decoder.update(mask=self.mask)
        hybrid_decoder.update(encode_value=self.encode_value)
        hybrid_decoder.update(return_feature=True)

        self.hybrid_decoder = build_decoder(hybrid_decoder)

        # init position decoder
        position_decoder.update(num_classes=self.num_classes)
        position_decoder.update(dim_input=self.dim_input)
        position_decoder.update(dim_model=self.dim_model)
        position_decoder.update(max_seq_len=self.max_seq_len)
        position_decoder.update(mask=self.mask)
        position_decoder.update(encode_value=self.encode_value)
        position_decoder.update(return_feature=True)

        self.position_decoder = build_decoder(position_decoder)

        self.fusion_module = RobustScannerFusionLayer(
            self.dim_model if encode_value else dim_input)

        pred_num_classes = num_classes - 1
        self.prediction = nn.Linear(dim_model if encode_value else dim_input,
                                    pred_num_classes)

    def forward_train(self, feat, out_enc, targets_dict, img_metas):
        hybrid_glimpse = self.hybrid_decoder.forward_train(
            feat, out_enc, targets_dict, img_metas)
        position_glimpse = self.position_decoder.forward_train(
            feat, out_enc, targets_dict, img_metas)

        fusion_out = self.fusion_module(hybrid_glimpse, position_glimpse)

        out = self.prediction(fusion_out)

        return out

    def forward_test(self, feat, out_enc, img_metas):
        seq_len = self.max_seq_len
        batch_size = feat.size(0)

        decode_sequence = (feat.new_ones(
            (batch_size, seq_len)) * self.start_idx).long()

        position_glimpse = self.position_decoder.forward_test(
            feat, out_enc, img_metas)

        outputs = []
        for i in range(seq_len):
            hybrid_glimpse_step = self.hybrid_decoder.forward_test_step(
                feat, out_enc, decode_sequence, i, img_metas)

            fusion_out = self.fusion_module(hybrid_glimpse_step,
                                            position_glimpse[:, i, :])

            char_out = self.prediction(fusion_out)
            char_out = F.softmax(char_out, -1)
            outputs.append(char_out)
            _, max_idx = torch.max(char_out, dim=1, keepdim=False)
            if i < seq_len - 1:
                decode_sequence[:, i + 1] = max_idx

        outputs = torch.stack(outputs, 1)

        return outputs
