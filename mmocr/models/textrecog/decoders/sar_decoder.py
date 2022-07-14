# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmocr.data import TextRecogDataSample
from mmocr.models.textrecog.dictionary import Dictionary
from mmocr.registry import MODELS
from .base_decoder import BaseDecoder


@MODELS.register_module()
class ParallelSARDecoder(BaseDecoder):
    """Implementation Parallel Decoder module in `SAR.

    <https://arxiv.org/abs/1811.00751>`_.

    Args:
        dictionary (dict or :obj:`Dictionary`): The config for `Dictionary` or
            the instance of `Dictionary`.
        module_loss (dict, optional): Config to build module_loss. Defaults
            to None.
        postprocessor (dict, optional): Config to build postprocessor.
            Defaults to None.
        enc_bi_rnn (bool): If True, use bidirectional RNN in encoder.
            Defaults to False.
        dec_bi_rnn (bool): If True, use bidirectional RNN in decoder.
            Defaults to False.
        dec_rnn_dropout (float): Dropout of RNN layer in decoder.
            Defaults to 0.0.
        dec_gru (bool): If True, use GRU, else LSTM in decoder. Defaults to
            False.
        d_model (int): Dim of channels from backbone :math:`D_i`. Defaults
            to 512.
        d_enc (int): Dim of encoder RNN layer :math:`D_m`. Defaults to 512.
        d_k (int): Dim of channels of attention module. Defaults to 64.
        pred_dropout (float): Dropout probability of prediction layer. Defaults
            to 0.0.
        max_seq_len (int): Maximum sequence length for decoding. Defaults to
            30.
        mask (bool): If True, mask padding in feature map. Defaults to True.
        pred_concat (bool): If True, concat glimpse feature from
            attention with holistic feature and hidden state. Defaults to
            False.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 dictionary: Union[Dict, Dictionary],
                 module_loss: Optional[Dict] = None,
                 postprocessor: Optional[Dict] = None,
                 enc_bi_rnn: bool = False,
                 dec_bi_rnn: bool = False,
                 dec_rnn_dropout: Union[int, float] = 0.0,
                 dec_gru: bool = False,
                 d_model: int = 512,
                 d_enc: int = 512,
                 d_k: int = 64,
                 pred_dropout: float = 0.0,
                 max_seq_len: int = 30,
                 mask: bool = True,
                 pred_concat: bool = False,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs) -> None:
        super().__init__(
            dictionary=dictionary,
            module_loss=module_loss,
            max_seq_len=max_seq_len,
            postprocessor=postprocessor,
            init_cfg=init_cfg)

        self.num_classes = self.dictionary.num_classes
        self.enc_bi_rnn = enc_bi_rnn
        self.d_k = d_k
        self.start_idx = self.dictionary.start_idx
        self.mask = mask
        self.pred_concat = pred_concat

        encoder_rnn_out_size = d_enc * (int(enc_bi_rnn) + 1)
        decoder_rnn_out_size = encoder_rnn_out_size * (int(dec_bi_rnn) + 1)
        # 2D attention layer
        self.conv1x1_1 = nn.Linear(decoder_rnn_out_size, d_k)
        self.conv3x3_1 = nn.Conv2d(
            d_model, d_k, kernel_size=3, stride=1, padding=1)
        self.conv1x1_2 = nn.Linear(d_k, 1)

        # Decoder RNN layer
        kwargs = dict(
            input_size=encoder_rnn_out_size,
            hidden_size=encoder_rnn_out_size,
            num_layers=2,
            batch_first=True,
            dropout=dec_rnn_dropout,
            bidirectional=dec_bi_rnn)
        if dec_gru:
            self.rnn_decoder = nn.GRU(**kwargs)
        else:
            self.rnn_decoder = nn.LSTM(**kwargs)

        # Decoder input embedding
        self.embedding = nn.Embedding(
            self.num_classes,
            encoder_rnn_out_size,
            padding_idx=self.dictionary.padding_idx)

        # Prediction layer
        self.pred_dropout = nn.Dropout(pred_dropout)
        if pred_concat:
            fc_in_channel = decoder_rnn_out_size + d_model + \
                            encoder_rnn_out_size
        else:
            fc_in_channel = d_model
        self.prediction = nn.Linear(fc_in_channel, self.num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def _2d_attention(self,
                      decoder_input: torch.Tensor,
                      feat: torch.Tensor,
                      holistic_feat: torch.Tensor,
                      valid_ratios: Optional[Sequence[float]] = None
                      ) -> torch.Tensor:
        """2D attention layer.

        Args:
            decoder_input (torch.Tensor): Input of decoder RNN.
            feat (torch.Tensor): Feature map of encoder.
            holistic_feat (torch.Tensor): Feature map of holistic encoder.
            valid_ratios (Sequence[float]): Valid ratios of attention.
                Defaults to None.

        Returns:
            torch.Tensor: Output of 2D attention layer.
        """
        y = self.rnn_decoder(decoder_input)[0]
        # y: bsz * (seq_len + 1) * hidden_size

        attn_query = self.conv1x1_1(y)  # bsz * (seq_len + 1) * attn_size
        bsz, seq_len, attn_size = attn_query.size()
        attn_query = attn_query.view(bsz, seq_len, attn_size, 1, 1)

        attn_key = self.conv3x3_1(feat)
        # bsz * attn_size * h * w
        attn_key = attn_key.unsqueeze(1)
        # bsz * 1 * attn_size * h * w

        attn_weight = torch.tanh(torch.add(attn_key, attn_query, alpha=1))
        # bsz * (seq_len + 1) * attn_size * h * w
        attn_weight = attn_weight.permute(0, 1, 3, 4, 2).contiguous()
        # bsz * (seq_len + 1) * h * w * attn_size
        attn_weight = self.conv1x1_2(attn_weight)
        # bsz * (seq_len + 1) * h * w * 1
        bsz, T, h, w, c = attn_weight.size()
        assert c == 1

        if valid_ratios is not None:
            # cal mask of attention weight
            attn_mask = torch.zeros_like(attn_weight)
            for i, valid_ratio in enumerate(valid_ratios):
                valid_width = min(w, math.ceil(w * valid_ratio))
                attn_mask[i, :, :, valid_width:, :] = 1
            attn_weight = attn_weight.masked_fill(attn_mask.bool(),
                                                  float('-inf'))

        attn_weight = attn_weight.view(bsz, T, -1)
        attn_weight = F.softmax(attn_weight, dim=-1)
        attn_weight = attn_weight.view(bsz, T, h, w,
                                       c).permute(0, 1, 4, 2, 3).contiguous()

        attn_feat = torch.sum(
            torch.mul(feat.unsqueeze(1), attn_weight), (3, 4), keepdim=False)
        # bsz * (seq_len + 1) * C

        # linear transformation
        if self.pred_concat:
            hf_c = holistic_feat.size(-1)
            holistic_feat = holistic_feat.expand(bsz, seq_len, hf_c)
            y = self.prediction(torch.cat((y, attn_feat, holistic_feat), 2))
        else:
            y = self.prediction(attn_feat)
        # bsz * (seq_len + 1) * num_classes
        y = self.pred_dropout(y)

        return y

    def forward_train(self, feat: torch.Tensor, out_enc: torch.Tensor,
                      data_samples: Sequence[TextRecogDataSample]
                      ) -> torch.Tensor:
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            data_samples (list[TextRecogDataSample]): Batch of
                TextRecogDataSample, containing gt_text and valid_ratio
                information.

        Returns:
            Tensor: A raw logit tensor of shape :math:`(N, T, C)`.
        """
        if data_samples is not None:
            assert len(data_samples) == feat.size(0)

        valid_ratios = [
            img_meta.get('valid_ratio', 1.0) for img_meta in data_samples
        ] if self.mask else None

        padded_targets = [
            data_sample.gt_text.padded_indexes for data_sample in data_samples
        ]
        padded_targets = torch.stack(padded_targets, dim=0).to(feat.device)
        tgt_embedding = self.embedding(padded_targets)
        # bsz * seq_len * emb_dim
        out_enc = out_enc.unsqueeze(1)
        # bsz * 1 * emb_dim
        in_dec = torch.cat((out_enc, tgt_embedding), dim=1)
        # bsz * (seq_len + 1) * C
        out_dec = self._2d_attention(
            in_dec, feat, out_enc, valid_ratios=valid_ratios)
        # bsz * (seq_len + 1) * num_classes

        return out_dec[:, 1:, :]  # bsz * seq_len * num_classes

    def forward_test(
        self,
        feat: torch.Tensor,
        out_enc: torch.Tensor,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            data_samples (list[TextRecogDataSample], optional): Batch of
                TextRecogDataSample, containing valid_ratio
                information. Defaults to None.

        Returns:
            Tensor: Character probabilities. of shape
            :math:`(N, self.max_seq_len, C)` where :math:`C` is
            ``num_classes``.
        """
        if data_samples is not None:
            assert len(data_samples) == feat.size(0)

        valid_ratios = None
        if data_samples is not None:
            valid_ratios = [
                data_sample.get('valid_ratio', 1.0)
                for data_sample in data_samples
            ] if self.mask else None

        seq_len = self.max_seq_len

        bsz = feat.size(0)
        start_token = torch.full((bsz, ),
                                 self.start_idx,
                                 device=feat.device,
                                 dtype=torch.long)
        # bsz
        start_token = self.embedding(start_token)
        # bsz * emb_dim
        start_token = start_token.unsqueeze(1).expand(-1, seq_len, -1)
        # bsz * seq_len * emb_dim
        out_enc = out_enc.unsqueeze(1)
        # bsz * 1 * emb_dim
        decoder_input = torch.cat((out_enc, start_token), dim=1)
        # bsz * (seq_len + 1) * emb_dim

        outputs = []
        for i in range(1, seq_len + 1):
            decoder_output = self._2d_attention(
                decoder_input, feat, out_enc, valid_ratios=valid_ratios)
            char_output = decoder_output[:, i, :]  # bsz * num_classes
            outputs.append(char_output)
            _, max_idx = torch.max(char_output, dim=1, keepdim=False)
            char_embedding = self.embedding(max_idx)  # bsz * emb_dim
            if i < seq_len:
                decoder_input[:, i + 1, :] = char_embedding

        outputs = torch.stack(outputs, 1)  # bsz * seq_len * num_classes

        return self.softmax(outputs)


@MODELS.register_module()
class SequentialSARDecoder(BaseDecoder):
    """Implementation Sequential Decoder module in `SAR.

    <https://arxiv.org/abs/1811.00751>`_.

    Args:
        dictionary (dict or :obj:`Dictionary`): The config for `Dictionary` or
            the instance of `Dictionary`.
        module_loss (dict, optional): Config to build module_loss. Defaults
            to None.
        postprocessor (dict, optional): Config to build postprocessor.
            Defaults to None.
        enc_bi_rnn (bool): If True, use bidirectional RNN in encoder. Defaults
            to False.
        dec_bi_rnn (bool): If True, use bidirectional RNN in decoder. Defaults
            to False.
        dec_do_rnn (float): Dropout of RNN layer in decoder. Defaults to 0.
        dec_gru (bool): If True, use GRU, else LSTM in decoder. Defaults to
            False.
        d_k (int): Dim of conv layers in attention module. Defaults to 64.
        d_model (int): Dim of channels from backbone :math:`D_i`. Defaults to
            512.
        d_enc (int): Dim of encoder RNN layer :math:`D_m`. Defaults to 512.
        pred_dropout (float): Dropout probability of prediction layer. Defaults
            to 0.
        max_seq_len (int): Maximum sequence length during decoding. Defaults to
            40.
        mask (bool): If True, mask padding in feature map. Defaults to False.
        pred_concat (bool): If True, concat glimpse feature from
            attention with holistic feature and hidden state. Defaults to
            False.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 dictionary: Optional[Union[Dict, Dictionary]] = None,
                 module_loss: Optional[Dict] = None,
                 postprocessor: Optional[Dict] = None,
                 enc_bi_rnn: bool = False,
                 dec_bi_rnn: bool = False,
                 dec_gru: bool = False,
                 d_k: int = 64,
                 d_model: int = 512,
                 d_enc: int = 512,
                 pred_dropout: float = 0.0,
                 mask: bool = True,
                 max_seq_len: int = 40,
                 pred_concat: bool = False,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs):
        super().__init__(
            dictionary=dictionary,
            module_loss=module_loss,
            postprocessor=postprocessor,
            max_seq_len=max_seq_len,
            init_cfg=init_cfg)

        self.num_classes = self.dictionary.num_classes
        self.enc_bi_rnn = enc_bi_rnn
        self.d_k = d_k
        self.start_idx = self.dictionary.start_idx
        self.dec_gru = dec_gru
        self.mask = mask
        self.pred_concat = pred_concat

        encoder_rnn_out_size = d_enc * (int(enc_bi_rnn) + 1)
        decoder_rnn_out_size = encoder_rnn_out_size * (int(dec_bi_rnn) + 1)
        # 2D attention layer
        self.conv1x1_1 = nn.Conv2d(
            decoder_rnn_out_size, d_k, kernel_size=1, stride=1)
        self.conv3x3_1 = nn.Conv2d(
            d_model, d_k, kernel_size=3, stride=1, padding=1)
        self.conv1x1_2 = nn.Conv2d(d_k, 1, kernel_size=1, stride=1)

        # Decoder rnn layer
        if dec_gru:
            self.rnn_decoder_layer1 = nn.GRUCell(encoder_rnn_out_size,
                                                 encoder_rnn_out_size)
            self.rnn_decoder_layer2 = nn.GRUCell(encoder_rnn_out_size,
                                                 encoder_rnn_out_size)
        else:
            self.rnn_decoder_layer1 = nn.LSTMCell(encoder_rnn_out_size,
                                                  encoder_rnn_out_size)
            self.rnn_decoder_layer2 = nn.LSTMCell(encoder_rnn_out_size,
                                                  encoder_rnn_out_size)

        # Decoder input embedding
        self.embedding = nn.Embedding(
            self.num_classes,
            encoder_rnn_out_size,
            padding_idx=self.dictionary.padding_idx)

        # Prediction layer
        self.pred_dropout = nn.Dropout(pred_dropout)
        if pred_concat:
            fc_in_channel = decoder_rnn_out_size + d_model + d_enc
        else:
            fc_in_channel = d_model
        self.prediction = nn.Linear(fc_in_channel, self.num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def _2d_attention(self,
                      y_prev: torch.Tensor,
                      feat: torch.Tensor,
                      holistic_feat: torch.Tensor,
                      hx1: torch.Tensor,
                      cx1: torch.Tensor,
                      hx2: torch.Tensor,
                      cx2: torch.Tensor,
                      valid_ratios: Optional[Sequence[float]] = None
                      ) -> torch.Tensor:
        """2D attention layer.

        Args:
            y_prev (torch.Tensor): Previous decoder hidden state.
            feat (torch.Tensor): Feature map.
            holistic_feat (torch.Tensor): Holistic feature map.
            hx1 (torch.Tensor): rnn decoder layer 1 hidden state.
            cx1 (torch.Tensor): rnn decoder layer 1 cell state.
            hx2 (torch.Tensor): rnn decoder layer 2 hidden state.
            cx2 (torch.Tensor): rnn decoder layer 2 cell state.
            valid_ratios (Optional[Sequence[float]]): Valid ratios of
                attention. Defaults to None.
        """
        _, _, h_feat, w_feat = feat.size()
        if self.dec_gru:
            hx1 = cx1 = self.rnn_decoder_layer1(y_prev, hx1)
            hx2 = cx2 = self.rnn_decoder_layer2(hx1, hx2)
        else:
            hx1, cx1 = self.rnn_decoder_layer1(y_prev, (hx1, cx1))
            hx2, cx2 = self.rnn_decoder_layer2(hx1, (hx2, cx2))

        tile_hx2 = hx2.view(hx2.size(0), hx2.size(1), 1, 1)
        attn_query = self.conv1x1_1(tile_hx2)  # bsz * attn_size * 1 * 1
        attn_query = attn_query.expand(-1, -1, h_feat, w_feat)
        attn_key = self.conv3x3_1(feat)
        attn_weight = torch.tanh(torch.add(attn_key, attn_query, alpha=1))
        attn_weight = self.conv1x1_2(attn_weight)
        bsz, c, h, w = attn_weight.size()
        assert c == 1

        if valid_ratios is not None:
            # cal mask of attention weight
            attn_mask = torch.zeros_like(attn_weight)
            for i, valid_ratio in enumerate(valid_ratios):
                valid_width = min(w, math.ceil(w * valid_ratio))
                attn_mask[i, :, :, valid_width:] = 1
            attn_weight = attn_weight.masked_fill(attn_mask.bool(),
                                                  float('-inf'))

        attn_weight = F.softmax(attn_weight.view(bsz, -1), dim=-1)
        attn_weight = attn_weight.view(bsz, c, h, w)

        attn_feat = torch.sum(
            torch.mul(feat, attn_weight), (2, 3), keepdim=False)  # n * c

        # linear transformation
        if self.pred_concat:
            y = self.prediction(torch.cat((hx2, attn_feat, holistic_feat), 1))
        else:
            y = self.prediction(attn_feat)

        return y, hx1, hx1, hx2, hx2

    def forward_train(
        self,
        feat: torch.Tensor,
        out_enc: torch.Tensor,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            data_samples (list[TextRecogDataSample]): Batch of
                TextRecogDataSample, containing gt_text and valid_ratio
                information.

        Returns:
            Tensor: A raw logit tensor of shape :math:`(N, T, C)`.
        """
        valid_ratios = None
        if data_samples is not None:
            valid_ratios = [
                data_sample.get('valid_ratio', 1.0)
                for data_sample in data_samples
            ] if self.mask else None

        padded_targets = [
            data_sample.gt_text.padded_indexes for data_sample in data_samples
        ]
        padded_targets = torch.stack(padded_targets, dim=0).to(feat.device)
        tgt_embedding = self.embedding(padded_targets)

        outputs = []
        for i in range(-1, self.max_seq_len):
            if i == -1:
                if self.dec_gru:
                    hx1 = cx1 = self.rnn_decoder_layer1(out_enc)
                    hx2 = cx2 = self.rnn_decoder_layer2(hx1)
                else:
                    hx1, cx1 = self.rnn_decoder_layer1(out_enc)
                    hx2, cx2 = self.rnn_decoder_layer2(hx1)
            else:
                y_prev = tgt_embedding[:, i, :]
                y, hx1, cx1, hx2, cx2 = self._2d_attention(
                    y_prev,
                    feat,
                    out_enc,
                    hx1,
                    cx1,
                    hx2,
                    cx2,
                    valid_ratios=valid_ratios)
                y = self.pred_dropout(y)

                outputs.append(y)

        outputs = torch.stack(outputs, 1)

        return outputs

    def forward_test(
        self,
        feat: torch.Tensor,
        out_enc: torch.Tensor,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            data_samples (list[TextRecogDataSample]): Batch of
                TextRecogDataSample, containing valid_ratio
                information.

        Returns:
            Tensor: Character probabilities. of shape
            :math:`(N, self.max_seq_len, C)` where :math:`C` is
            ``num_classes``.
        """
        valid_ratios = None
        if data_samples is not None:
            valid_ratios = [
                data_sample.get('valid_ratio', 1.0)
                for data_sample in data_samples
            ] if self.mask else None

        outputs = []
        start_token = torch.full((feat.size(0), ),
                                 self.start_idx,
                                 device=feat.device,
                                 dtype=torch.long)
        start_token = self.embedding(start_token)
        for i in range(-1, self.max_seq_len):
            if i == -1:
                if self.dec_gru:
                    hx1 = cx1 = self.rnn_decoder_layer1(out_enc)
                    hx2 = cx2 = self.rnn_decoder_layer2(hx1)
                else:
                    hx1, cx1 = self.rnn_decoder_layer1(out_enc)
                    hx2, cx2 = self.rnn_decoder_layer2(hx1)
                    y_prev = start_token
            else:
                y, hx1, cx1, hx2, cx2 = self._2d_attention(
                    y_prev,
                    feat,
                    out_enc,
                    hx1,
                    cx1,
                    hx2,
                    cx2,
                    valid_ratios=valid_ratios)
                _, max_idx = torch.max(y, dim=1, keepdim=False)
                char_embedding = self.embedding(max_idx)
                y_prev = char_embedding
                outputs.append(y)

        outputs = torch.stack(outputs, 1)

        return self.softmax(outputs)
