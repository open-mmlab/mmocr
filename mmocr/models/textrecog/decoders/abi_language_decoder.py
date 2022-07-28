# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import BaseTransformerLayer
from mmengine.model import ModuleList

from mmocr.data import TextRecogDataSample
from mmocr.models.common.modules import PositionalEncoding
from mmocr.models.textrecog.dictionary import Dictionary
from mmocr.registry import MODELS
from .base_decoder import BaseDecoder


@MODELS.register_module()
class ABILanguageDecoder(BaseDecoder):
    r"""Transformer-based language model responsible for spell correction.
    Implementation of language model of \
        `ABINet <https://arxiv.org/abs/1910.04396>`_.

    Args:
        dictionary (dict or :obj:`Dictionary`): The config for `Dictionary` or
            the instance of `Dictionary`. The dictionary must have an end
            token.
        d_model (int): Hidden size :math:`E` of model. Defaults to 512.
        n_head (int): Number of multi-attention heads.
        d_inner (int): Hidden size of feedforward network model.
        n_layers (int): The number of similar decoding layers.
        dropout (float): Dropout rate.
        detach_tokens (bool): Whether to block the gradient flow at input
         tokens.
        use_self_attn (bool): If True, use self attention in decoder layers,
            otherwise cross attention will be used.
        max_seq_len (int): Maximum sequence length :math:`T`. The
            sequence is usually generated from decoder. Defaults to 40.
        module_loss (dict, optional): Config to build loss. Defaults to None.
        postprocessor (dict, optional): Config to build postprocessor.
            Defaults to None.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 dictionary: Union[Dict, Dictionary],
                 d_model: int = 512,
                 n_head: int = 8,
                 d_inner: int = 2048,
                 n_layers: int = 4,
                 dropout: float = 0.1,
                 detach_tokens: bool = True,
                 use_self_attn: bool = False,
                 max_seq_len: int = 40,
                 module_loss: Optional[Dict] = None,
                 postprocessor: Optional[Dict] = None,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs) -> None:
        super().__init__(
            dictionary=dictionary,
            module_loss=module_loss,
            postprocessor=postprocessor,
            max_seq_len=max_seq_len,
            init_cfg=init_cfg)

        assert self.dictionary.end_idx is not None,\
            'Dictionary must contain an end token! (with_end=True)'

        self.detach_tokens = detach_tokens
        self.d_model = d_model

        self.proj = nn.Linear(self.dictionary.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(
            d_model, n_position=self.max_seq_len, dropout=0.1)
        self.pos_encoder = PositionalEncoding(
            d_model, n_position=self.max_seq_len)

        if use_self_attn:
            operation_order = ('self_attn', 'norm', 'cross_attn', 'norm',
                               'ffn', 'norm')
        else:
            operation_order = ('cross_attn', 'norm', 'ffn', 'norm')

        decoder_layer = BaseTransformerLayer(
            operation_order=operation_order,
            attn_cfgs=dict(
                type='MultiheadAttention',
                embed_dims=d_model,
                num_heads=n_head,
                attn_drop=dropout,
                dropout_layer=dict(type='Dropout', drop_prob=dropout),
            ),
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=d_model,
                feedforward_channels=d_inner,
                ffn_drop=dropout,
            ),
            norm_cfg=dict(type='LN'),
        )
        self.decoder_layers = ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(n_layers)])

        self.cls = nn.Linear(d_model, self.dictionary.num_classes)

    def forward_train(
            self,
            feat: Optional[torch.Tensor] = None,
            out_enc: torch.Tensor = None,
            data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> Dict:
        """
        Args:
            feat (torch.Tensor, optional): Not required. Feature map
                placeholder. Defaults to None.
            out_enc (torch.Tensor): Logits with shape :math:`(N, T, C)`.
                Defaults to None.
            data_samples (list[TextRecogDataSample], optional): Not required.
                DataSample placeholder. Defaults to None.

        Returns:
            A dict with keys ``feature`` and ``logits``.

            - feature (Tensor): Shape :math:`(N, T, E)`. Raw textual features
              for vision language aligner.
            - logits (Tensor): Shape :math:`(N, T, C)`. The raw logits for
              characters after spell correction.
        """
        lengths = self._get_length(out_enc)
        lengths.clamp_(2, self.max_seq_len)
        tokens = torch.softmax(out_enc, dim=-1)
        if self.detach_tokens:
            tokens = tokens.detach()
        embed = self.proj(tokens)  # (N, T, E)
        embed = self.token_encoder(embed)  # (N, T, E)
        padding_mask = self._get_padding_mask(lengths, self.max_seq_len)

        zeros = embed.new_zeros(*embed.shape)
        query = self.pos_encoder(zeros)
        query = query.permute(1, 0, 2)  # (T, N, E)
        embed = embed.permute(1, 0, 2)
        location_mask = self._get_location_mask(self.max_seq_len,
                                                tokens.device)
        output = query
        for m in self.decoder_layers:
            output = m(
                query=output,
                key=embed,
                value=embed,
                attn_masks=location_mask,
                key_padding_mask=padding_mask)
        output = output.permute(1, 0, 2)  # (N, T, E)

        out_enc = self.cls(output)  # (N, T, C)
        return {'feature': output, 'logits': out_enc}

    def forward_test(
            self,
            feat: Optional[torch.Tensor] = None,
            logits: torch.Tensor = None,
            data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> Dict:
        """
        Args:
            feat (torch.Tensor, optional): Not required. Feature map
                placeholder. Defaults to None.
            logits (Tensor): Raw language logitis. Shape :math:`(N, T, C)`.
                Defaults to None.
            data_samples (list[TextRecogDataSample], optional): Not required.
                DataSample placeholder. Defaults to None.

        Returns:
            A dict with keys ``feature`` and ``logits``.

            - feature (Tensor): Shape :math:`(N, T, E)`. Raw textual features
              for vision language aligner.
            - logits (Tensor): Shape :math:`(N, T, C)`. The raw logits for
              characters after spell correction.
        """
        return self.forward_train(feat, logits, data_samples)

    def _get_length(self, logit: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Greedy decoder to obtain length from logit.

        Returns the first location of padding index or the length of the entire
        tensor otherwise.
        """
        # out as a boolean vector indicating the existence of end token(s)
        out = (logit.argmax(dim=-1) == self.dictionary.end_idx)
        abn = out.any(dim)
        # Get the first index of end token
        out = ((out.cumsum(dim) == 1) & out).max(dim)[1]
        out = out + 1
        out = torch.where(abn, out, out.new_tensor(logit.shape[1]))
        return out

    @staticmethod
    def _get_location_mask(seq_len: int,
                           device: Union[Optional[torch.device],
                                         str] = None) -> torch.Tensor:
        """Generate location masks given input sequence length.

        Args:
            seq_len (int): The length of input sequence to transformer.
            device (torch.device or str, optional): The device on which the
                masks will be placed.

        Returns:
            Tensor: A mask tensor of shape (seq_len, seq_len) with -infs on
            diagonal and zeros elsewhere.
        """
        mask = torch.eye(seq_len, device=device)
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        return mask

    @staticmethod
    def _get_padding_mask(length: int, max_length: int) -> torch.Tensor:
        """Generate padding masks.

        Args:
            length (Tensor): Shape :math:`(N,)`.
            max_length (int): The maximum sequence length :math:`T`.

        Returns:
            Tensor: A bool tensor of shape :math:`(N, T)` with Trues on
            elements located over the length, or Falses elsewhere.
        """
        length = length.unsqueeze(-1)
        grid = torch.arange(0, max_length, device=length.device).unsqueeze(0)
        return grid >= length
