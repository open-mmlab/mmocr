import torch
import torch.nn as nn
import torch.nn.functional as F

from mmocr.models.builder import DECODERS
from mmocr.models.textrecog.layers import (DecoderLayer, PositionalEncoding,
                                           get_pad_mask, get_subsequent_mask)
from .base_decoder import BaseDecoder


@DECODERS.register_module()
class TFDecoder(BaseDecoder):
    """Transformer Decoder block with self attention mechanism."""

    def __init__(self,
                 n_layers=6,
                 d_embedding=512,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 d_model=512,
                 d_inner=256,
                 n_position=200,
                 dropout=0.1,
                 num_classes=93,
                 max_seq_len=40,
                 start_idx=1,
                 padding_idx=92,
                 **kwargs):
        super().__init__()

        self.padding_idx = padding_idx
        self.start_idx = start_idx
        self.max_seq_len = max_seq_len

        self.trg_word_emb = nn.Embedding(
            num_classes, d_embedding, padding_idx=padding_idx)

        self.position_enc = PositionalEncoding(
            d_embedding, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        pred_num_class = num_classes - 1  # ignore padding_idx
        self.classifier = nn.Linear(d_model, pred_num_class)

    def _attention(self, trg_seq, src, src_mask=None):
        trg_embedding = self.trg_word_emb(trg_seq)
        trg_pos_encoded = self.position_enc(trg_embedding)
        tgt = self.dropout(trg_pos_encoded)

        trg_mask = get_pad_mask(
            trg_seq, pad_idx=self.padding_idx) & get_subsequent_mask(trg_seq)
        output = tgt
        for dec_layer in self.layer_stack:
            output = dec_layer(
                output,
                src,
                slf_attn_mask=trg_mask,
                dec_enc_attn_mask=src_mask)
        output = self.layer_norm(output)

        return output

    def forward_train(self, feat, out_enc, targets_dict, img_metas):
        targets = targets_dict['padded_targets'].to(out_enc.device)
        attn_output = self._attention(targets, out_enc, src_mask=None)
        outputs = self.classifier(attn_output)
        return outputs

    def forward_test(self, feat, out_enc, img_metas):
        bsz = out_enc.size(0)
        init_target_seq = torch.full((bsz, self.max_seq_len + 1),
                                     self.padding_idx,
                                     device=out_enc.device,
                                     dtype=torch.long)
        # bsz * seq_len
        init_target_seq[:, 0] = self.start_idx

        outputs = []
        for step in range(0, self.max_seq_len):
            decoder_output = self._attention(
                init_target_seq, out_enc, src_mask=None)
            # bsz * seq_len * 512
            step_result = F.softmax(
                self.classifier(decoder_output[:, step, :]), dim=-1)
            # bsz * num_classes
            outputs.append(step_result)
            _, step_max_index = torch.max(step_result, dim=-1)
            init_target_seq[:, step + 1] = step_max_index

        outputs = torch.stack(outputs, dim=1)

        return outputs
