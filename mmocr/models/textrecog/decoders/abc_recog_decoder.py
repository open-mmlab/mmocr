# Copyright (c) OpenMMLab. All rights reserved.
import random

import torch

from mmocr.models.builder import DECODERS
from mmocr.models.textrecog.layers import AttentionGRUCell
from .base_decoder import BaseDecoder


@DECODERS.register_module()
class ABCRecogDecoder(BaseDecoder):
    """Decoder for ABCNet's recognition branch.

    Args:
        num_channels (int): Number of channels of hidden vectors :math:`E`.
        teacher_forcing (float): The prbobability of enabling the teacher
            forcing strategy during training.
        max_seq_len (int): Maximum text sequence length :math:`T`.
        num_chars (int): Number of text characters :math:`C`.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels=None,
                 num_chars=None,
                 max_seq_len=None,
                 teacher_forcing=0.5,
                 dropout=0.1,
                 init_cfg=dict(type='Xavier', layer='Conv2d'),
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.num_chars = num_chars
        self.max_seq_len = max_seq_len
        self.teacher_forcing = teacher_forcing
        self.attention_cell = AttentionGRUCell(in_channels, num_chars, dropout)

    def forward_train(self, feat, out_enc, targets_dict, img_metas):
        """
        Args:
            feat (Tensor): Unused.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, T_e, C)`.
            targets_dict (dict): A dict with the key ``padded_targets``, a
                tensor of shape :math:`(N, T)`. Each element is the index of a
                character.
            img_metas (dict): Unused.

        Returns:
            Tensor: A raw logit tensor of shape :math:`(N, T, C)`.
        """
        return self._decode(out_enc, targets_dict['padded_targets'])

    def forward_test(self, feat, out_enc, img_metas):
        """
        Args:
            feat (Tensor): Unused.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, T_e, C)`.
            img_metas (dict): Unused.

        Returns:
            Tensor: A raw logit tensor of shape :math:`(N, T, C)`.
        """
        return self._decode(out_enc)

    def _decode(self, out_enc, target=None):
        """
        Args:
            out_enc (Tensor): Encoder output of shape
                :math:`(N, T_e, C)`.
            targets_dict (dict): A tensor of shape :math:`(N, T)`. Each element
                is the index of a character.

        Returns:
            Tensor: A raw logit tensor of shape :math:`(N, T, C)`.
        """

        N = out_enc.shape[0]
        out_enc = out_enc.permute(1, 0, 2)  # (W, N, C)
        testing = target is None
        T = self.max_seq_len
        if not testing:
            T = min(T, target.shape[1])

        # Should be SOS
        decoder_input = torch.zeros((N, 1),
                                    dtype=torch.long,
                                    device=out_enc.device)

        decoder_hidden = self.attention_cell.init_hidden(N)
        results = []
        for i in range(T):
            decoder_output, decoder_hidden, _ = self.attention_cell(
                decoder_input, decoder_hidden, out_enc)
            results.append(decoder_output)
            teacher_focing = False if testing else \
                random.random() < self.teacher_forcing
            if teacher_focing:
                decoder_input = target[:, i]
            else:
                _, top_idxs = decoder_output.data.topk(1)
                decoder_input = top_idxs
        return torch.stack(results, dim=1)
