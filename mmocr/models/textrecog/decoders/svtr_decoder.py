# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmocr.models.common.dictionary import Dictionary
from mmocr.registry import MODELS
from mmocr.structures import TextRecogDataSample
from .base import BaseDecoder


@MODELS.register_module()
class SVTRDecoder(BaseDecoder):
    """Decoder module in `SVTR <https://arxiv.org/abs/2205.00159>`_.

    Args:
        in_channels (int): The num of input channels.
        dictionary (Union[Dict, Dictionary]): The config for `Dictionary` or
            the instance of `Dictionary`. Defaults to None.
        module_loss (Optional[Dict], optional): Cfg to build module_loss.
            Defaults to None.
        postprocessor (Optional[Dict], optional): Cfg to build postprocessor.
            Defaults to None.
        max_seq_len (int, optional): Maximum output sequence length :math:`T`.
            Defaults to 25.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 dictionary: Union[Dict, Dictionary] = None,
                 module_loss: Optional[Dict] = None,
                 postprocessor: Optional[Dict] = None,
                 max_seq_len: int = 25,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:

        super().__init__(
            dictionary=dictionary,
            module_loss=module_loss,
            postprocessor=postprocessor,
            max_seq_len=max_seq_len,
            init_cfg=init_cfg)

        self.in_channels = in_channels
        self.decoder = nn.Linear(
            in_features=in_channels, out_features=self.dictionary.num_classes)
        # self.log_softmax =
        self.apply(self._init_weights)
        print('-----------decoder weight inits-------------------')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            stdv = 1.0 / math.sqrt(self.in_channels * 1.0)
            nn.init.uniform_(m.weight, -stdv, stdv)
            nn.init.uniform_(m.bias, -stdv, stdv)

    def forward_train(
        self,
        feat: Optional[torch.Tensor] = None,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        """Forward for training.

        Args:
            feat (torch.Tensor, optional): The feature map from backbone of
                shape :math:`(N, E, H, W)`. Defaults to None.
            out_enc (torch.Tensor, optional): Encoder output. Defaults to None.
            data_samples (Sequence[TextRecogDataSample]): Batch of
                TextRecogDataSample, containing gt_text information. Defaults
                to None.

        Returns:
            Tensor: The raw logit tensor. Shape :math:`(N, T, C)` where
            :math:`C` is ``num_classes``.
        """
        assert feat.size(2) == 1, 'feature height must be 1'
        x = feat.squeeze(2)
        x = x.permute(0, 2, 1)
        predicts = self.decoder(x)
        return predicts

    def forward_test(
        self,
        feat: Optional[torch.Tensor] = None,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        """Forward for testing.

        Args:
            feat (torch.Tensor, optional): The feature map from backbone of
                shape :math:`(N, E, H, W)`. Defaults to None.
            out_enc (torch.Tensor, optional): Encoder output. Defaults to None.
            data_samples (Sequence[TextRecogDataSample]): Batch of
                TextRecogDataSample, containing gt_text information. Defaults
                to None.

        Returns:
            Tensor: Character probabilities. of shape
            :math:`(self.max_seq_len, N, C)` where :math:`C` is
            ``num_classes``.
        """
        feat = self.forward_train(feat, out_enc, data_samples).permute(1, 0, 2)
        return F.log_softmax(feat, dim=2).requires_grad_()
