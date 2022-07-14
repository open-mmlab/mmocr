# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

from mmocr.data import TextRecogDataSample
from mmocr.models.textrecog.dictionary import Dictionary
from mmocr.registry import MODELS
from .base_decoder import BaseDecoder


@MODELS.register_module()
class RobustScannerFuser(BaseDecoder):
    """Decoder for RobustScanner.

    Args:
        dictionary (dict or :obj:`Dictionary`): The config for `Dictionary` or
            the instance of `Dictionary`.
        module_loss (dict, optional): Config to build module_loss. Defaults
            to None.
        postprocessor (dict, optional): Config to build postprocessor.
            Defaults to None.
        hybrid_decoder (dict): Config to build hybrid_decoder. Defaults to
            dict(type='SequenceAttentionDecoder').
        position_decoder (dict): Config to build position_decoder. Defaults to
            dict(type='PositionAttentionDecoder').
        fuser (dict): Config to build fuser. Defaults to
            dict(type='RobustScannerFuser').
        max_seq_len (int): Maximum sequence length. The
            sequence is usually generated from decoder. Defaults to 30.
        in_channels (list[int]): List of input channels.
            Defaults to [512, 512].
        dim (int): The dimension on which to split the input. Defaults to -1.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 dictionary: Union[Dict, Dictionary],
                 module_loss: Optional[Dict] = None,
                 postprocessor: Optional[Dict] = None,
                 hybrid_decoder: Dict = dict(type='SequenceAttentionDecoder'),
                 position_decoder: Dict = dict(
                     type='PositionAttentionDecoder'),
                 max_seq_len: int = 30,
                 in_channels: List[int] = [512, 512],
                 dim: int = -1,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(
            dictionary=dictionary,
            module_loss=module_loss,
            postprocessor=postprocessor,
            max_seq_len=max_seq_len,
            init_cfg=init_cfg)

        for cfg_name in ['hybrid_decoder', 'position_decoder']:
            cfg = eval(cfg_name)
            if cfg is not None:
                if cfg.get('dictionary', None) is None:
                    cfg.update(dictionary=self.dictionary)
                else:
                    warnings.warn(f"Using dictionary {cfg['dictionary']} "
                                  "in decoder's config.")
                if cfg.get('max_seq_len', None) is None:
                    cfg.update(max_seq_len=max_seq_len)
                else:
                    warnings.warn(f"Using max_seq_len {cfg['max_seq_len']} "
                                  "in decoder's config.")
                setattr(self, cfg_name, MODELS.build(cfg))

        in_channels = sum(in_channels)
        self.dim = dim

        self.linear_layer = nn.Linear(in_channels, in_channels)
        self.glu_layer = nn.GLU(dim=dim)
        self.prediction = nn.Linear(
            int(in_channels / 2), self.dictionary.num_classes)
        self.softmax = nn.Softmax(dim=-1)

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
        """
        hybrid_glimpse = self.hybrid_decoder(feat, out_enc, data_samples)
        position_glimpse = self.position_decoder(feat, out_enc, data_samples)
        fusion_input = torch.cat([hybrid_glimpse, position_glimpse], self.dim)
        outputs = self.linear_layer(fusion_input)
        outputs = self.glu_layer(outputs)
        return self.prediction(outputs)

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
                TextRecogDataSample, containing vaild_ratio information.
                Defaults to None.

        Returns:
            Tensor: Character probabilities. of shape
            :math:`(N, self.max_seq_len, C)` where :math:`C` is
            ``num_classes``.
        """
        position_glimpse = self.position_decoder(feat, out_enc, data_samples)

        batch_size = feat.size(0)
        decode_sequence = (feat.new_ones((batch_size, self.max_seq_len)) *
                           self.dictionary.start_idx).long()
        outputs = []
        for step in range(self.max_seq_len):
            hybrid_glimpse_step = self.hybrid_decoder.forward_test_step(
                feat, out_enc, decode_sequence, step, data_samples)

            fusion_input = torch.cat(
                [hybrid_glimpse_step, position_glimpse[:, step, :]], self.dim)
            output = self.linear_layer(fusion_input)
            output = self.glu_layer(output)
            output = self.prediction(output)
            _, max_idx = torch.max(output, dim=1, keepdim=False)
            if step < self.max_seq_len - 1:
                decode_sequence[:, step + 1] = max_idx
            outputs.append(output)
        outputs = torch.stack(outputs, 1)
        return self.softmax(outputs)
