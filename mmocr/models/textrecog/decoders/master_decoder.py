# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import BaseTransformerLayer
from mmcv.runner import ModuleList

from mmocr.core.data_structures import TextRecogDataSample
from mmocr.models.common.modules import PositionalEncoding
from mmocr.models.textrecog.dictionary import Dictionary
from mmocr.registry import MODELS
from .base_decoder import BaseDecoder


def clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Produce N identical layers.

    Args:
        module (nn.Module): A pytorch nn.module.
        N (int): Number of copies.

    Returns:
        nn.ModuleList: A pytorch nn.ModuleList with the copies.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Embeddings(nn.Module):
    """Construct the word embeddings given vocab size and embed dim.

    Args:
        d_model (int): The embedding dimension.
        vocab (int): Vocablury size.
    """

    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, *input: torch.Tensor) -> torch.Tensor:
        """Forward the embeddings.

        Args:
            input (torch.Tensor): The input tensors.

        Returns:
            torch.Tensor: The embeddings.
        """
        x = input[0]
        return self.lut(x) * math.sqrt(self.d_model)


@MODELS.register_module()
class MasterDecoder(BaseDecoder):
    """Decoder module in `MASTER <https://arxiv.org/abs/1910.02562>`_.

    Code is partially modified from https://github.com/wenwenyu/MASTER-pytorch.

    Args:
        n_layers (int): Number of attention layers.
        n_head (int): Number of parallel attention heads.
        d_model (int): Dimension :math:`E` of the input from previous model.
        feat_size (int): The size of the input feature from previous model,
            usually :math:`H * W`.
        d_inner (int): Hidden dimension of feedforward layers.
        attn_drop (float): Dropout rate of the attention layer.
        ffn_drop (float): Dropout rate of the feedforward layer.
        feat_pe_drop (float): Dropout rate of the feature positional encoding
            layer.
        dictionary (dict or :obj:`Dictionary`): The config for `Dictionary` or
            the instance of `Dictionary`.
        loss (dict, optional): Config to build loss. Defaults to None.
        postprocessor (dict, optional): Config to build postprocessor.
            Defaults to None.
        max_seq_len (int): Maximum output sequence length :math:`T`.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(
        self,
        n_layers: int = 3,
        n_head: int = 8,
        d_model: int = 512,
        feat_size: int = 6 * 40,
        d_inner: int = 2048,
        attn_drop: float = 0.,
        ffn_drop: float = 0.,
        feat_pe_drop: float = 0.2,
        loss: Optional[Dict] = None,
        postprocessor: Optional[Dict] = None,
        dictionary: Optional[Union[Dict, Dictionary]] = None,
        max_seq_len: int = 30,
        init_cfg: Optional[Union[Dict, Sequence[Dict]]] = None,
    ):
        super().__init__(
            loss=loss,
            postprocessor=postprocessor,
            dictionary=dictionary,
            init_cfg=init_cfg,
            max_seq_len=max_seq_len)
        operation_order = ('norm', 'self_attn', 'norm', 'cross_attn', 'norm',
                           'ffn')
        decoder_layer = BaseTransformerLayer(
            operation_order=operation_order,
            attn_cfgs=dict(
                type='MultiheadAttention',
                embed_dims=d_model,
                num_heads=n_head,
                attn_drop=attn_drop,
                dropout_layer=dict(type='Dropout', drop_prob=attn_drop),
            ),
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=d_model,
                feedforward_channels=d_inner,
                ffn_drop=ffn_drop,
                dropout_layer=dict(type='Dropout', drop_prob=ffn_drop),
            ),
            norm_cfg=dict(type='LN'),
            batch_first=True,
        )
        self.decoder_layers = ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(n_layers)])

        self.cls = nn.Linear(d_model, self.dictionary.num_classes)

        self.SOS = self.dictionary.start_idx
        self.PAD = self.dictionary.padding_idx
        self.max_seq_len = max_seq_len
        self.feat_size = feat_size
        self.n_head = n_head

        self.embedding = Embeddings(
            d_model=d_model, vocab=self.dictionary.num_classes)

        # TODO:
        self.positional_encoding = PositionalEncoding(
            d_hid=d_model, n_position=self.max_seq_len + 1)
        self.feat_positional_encoding = PositionalEncoding(
            d_hid=d_model, n_position=self.feat_size, dropout=feat_pe_drop)
        self.norm = nn.LayerNorm(d_model)

    def make_target_mask(self, tgt: torch.Tensor,
                         device: torch.device) -> torch.Tensor:
        """Make target mask for self attention.

        Args:
            tgt (Tensor): Shape [N, l_tgt]
            device (torch.device): Mask device.

        Returns:
            Tensor: Mask of shape [N * self.n_head, l_tgt, l_tgt]
        """

        trg_pad_mask = (tgt != self.PAD).unsqueeze(1).unsqueeze(3).bool()
        tgt_len = tgt.size(1)
        trg_sub_mask = torch.tril(
            torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=device))
        tgt_mask = trg_pad_mask & trg_sub_mask

        # inverse for mmcv's BaseTransformerLayer
        tril_mask = tgt_mask.clone()
        tgt_mask = tgt_mask.float().masked_fill_(tril_mask == 0, -1e9)
        tgt_mask = tgt_mask.masked_fill_(tril_mask, 0)
        tgt_mask = tgt_mask.repeat(1, self.n_head, 1, 1)
        tgt_mask = tgt_mask.view(-1, tgt_len, tgt_len)
        return tgt_mask

    def decode(self, tgt_seq: torch.Tensor, feature: torch.Tensor,
               src_mask: torch.BoolTensor,
               tgt_mask: torch.BoolTensor) -> torch.Tensor:
        """Decode the input sequence.

        Args:
            tgt_seq (Tensor): Target sequence of shape: math: `(N, T, C)`.
            feature (Tensor): Input feature map from encoder of
                shape: math: `(N, C, H, W)`
            src_mask (BoolTensor): The source mask of shape: math: `(N, H*W)`.
            tgt_mask (BoolTensor): The target mask of shape: math: `(N, T, T)`.

        Return:
            Tensor: The decoded sequence.
        """
        tgt_seq = self.embedding(tgt_seq)
        x = self.positional_encoding(tgt_seq)
        attn_masks = [tgt_mask, src_mask]
        for layer in self.decoder_layers:
            x = layer(
                query=x, key=feature, value=feature, attn_masks=attn_masks)
        x = self.norm(x)
        return self.cls(x)

    def forward_train(self,
                      feat: Optional[torch.Tensor] = None,
                      out_enc: torch.Tensor = None,
                      data_samples: Sequence[TextRecogDataSample] = None
                      ) -> torch.Tensor:
        """Forward for training. Source mask will not be used here.

        Args:
            feat (Tensor, optional): Input feature map from backbone.
            out_enc (Tensor): Unused.
            data_samples (list[TextRecogDataSample]): Batch of
                TextRecogDataSample, containing gt_text and valid_ratio
                information.

        Returns:
            Tensor: The raw logit tensor. Shape :math:`(N, T, C)` where
            :math:`C` is ``num_classes``.
        """

        # flatten 2D feature map
        if len(feat.shape) > 3:
            b, c, h, w = feat.shape
            feat = feat.view(b, c, h * w)
            feat = feat.permute((0, 2, 1))
        feat = self.feat_positional_encoding(feat)

        targets = self.loss.get_targets(data_samples)
        trg_seq = []
        for target in targets:
            trg_seq.append(target.gt_text.padded_indexes.to(feat.device))

        trg_seq = torch.stack(trg_seq, dim=0)

        src_mask = None
        tgt_mask = self.make_target_mask(trg_seq, device=feat.device)
        return self.decode(trg_seq, feat, src_mask, tgt_mask)

    def forward_test(self,
                     feat: Optional[torch.Tensor] = None,
                     out_enc: torch.Tensor = None,
                     data_samples: Sequence[TextRecogDataSample] = None
                     ) -> torch.Tensor:
        """Forward for testing.

        Args:
            feat (Tensor, optional): Input feature map from backbone.
            out_enc (Tensor): Unused.
            data_samples (list[TextRecogDataSample]): Unused.

        Returns:
            Tensor: The raw logit tensor.
            Shape :math:`(N, self.max_seq_len, C)` where :math:`C` is
            ``num_classes``.
        """

        # flatten 2D feature map
        if len(feat.shape) > 3:
            b, c, h, w = feat.shape
            feat = feat.view(b, c, h * w)
            feat = feat.permute((0, 2, 1))
        feat = self.feat_positional_encoding(feat)

        N = feat.shape[0]
        input = torch.full((N, 1),
                           self.SOS,
                           device=feat.device,
                           dtype=torch.long)
        output = None
        for _ in range(self.max_seq_len):
            target_mask = self.make_target_mask(input, device=feat.device)
            out = self.decode(input, feat, None, target_mask)
            output = out
            prob = F.softmax(out, dim=-1)
            _, next_word = torch.max(prob, dim=-1)
            input = torch.cat([input, next_word[:, -1].unsqueeze(-1)], dim=1)
        return output
