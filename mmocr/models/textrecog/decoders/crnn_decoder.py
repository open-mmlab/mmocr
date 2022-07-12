# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
from mmcv.runner import Sequential

from mmocr.data import TextRecogDataSample
from mmocr.models.textrecog.dictionary import Dictionary
from mmocr.models.textrecog.layers import BidirectionalLSTM
from mmocr.registry import MODELS
from .base_decoder import BaseDecoder


@MODELS.register_module()
class CRNNDecoder(BaseDecoder):
    """Decoder for CRNN.

    Args:
        in_channels (int): Number of input channels.
        dictionary (dict or :obj:`Dictionary`): The config for `Dictionary` or
            the instance of `Dictionary`.
        rnn_flag (bool): Use RNN or CNN as the decoder. Defaults to False.
        loss_module (dict, optional): Config to build loss_module. Defaults
            to None.
        postprocessor (dict, optional): Config to build postprocessor.
            Defaults to None.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 dictionary: Union[Dictionary, Dict],
                 rnn_flag: bool = False,
                 loss_module: Dict = None,
                 postprocessor: Dict = None,
                 init_cfg=dict(type='Xavier', layer='Conv2d'),
                 **kwargs):
        super().__init__(
            init_cfg=init_cfg,
            dictionary=dictionary,
            loss_module=loss_module,
            postprocessor=postprocessor)
        self.rnn_flag = rnn_flag

        if rnn_flag:
            self.decoder = Sequential(
                BidirectionalLSTM(in_channels, 256, 256),
                BidirectionalLSTM(256, 256, self.dictionary.num_classes))
        else:
            self.decoder = nn.Conv2d(
                in_channels,
                self.dictionary.num_classes,
                kernel_size=1,
                stride=1)

    def forward_train(
        self,
        feat: torch.Tensor,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        """
        Args:
            feat (Tensor): A Tensor of shape :math:`(N, C, 1, W)`.
            out_enc (torch.Tensor, optional): Encoder output. Defaults to None.
            data_samples (list[TextRecogDataSample], optional): Batch of
                TextRecogDataSample, containing gt_text information. Defaults
                to None.

        Returns:
            Tensor: The raw logit tensor. Shape :math:`(N, W, C)` where
            :math:`C` is ``num_classes``.
        """
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

    def forward_test(
        self,
        feat: Optional[torch.Tensor] = None,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        """
        Args:
            feat (Tensor): A Tensor of shape :math:`(N, C, 1, W)`.
            out_enc (torch.Tensor, optional): Encoder output. Defaults to None.
            data_samples (list[TextRecogDataSample]): Batch of
                TextRecogDataSample, containing ``gt_text`` information.
                Defaults to None.

        Returns:
            Tensor: The raw logit tensor. Shape :math:`(N, W, C)` where
            :math:`C` is ``num_classes``.
        """
        return self.forward_train(feat, out_enc, data_samples)
