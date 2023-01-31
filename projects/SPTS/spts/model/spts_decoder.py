import copy
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmocr.models.common import Dictionary
from mmocr.models.textrecog.decoders import BaseDecoder
from mmocr.registry import MODELS
from mmocr.utils.typing_utils import TextSpottingDataSample
from .position_embedding import PositionEmbeddingSine


@MODELS.register_module()
class SPTSDecoder(BaseDecoder):
    """SPTS Decoder.

    Args:
        dictionary (dict or :obj:`Dictionary`): The config for `Dictionary` or
            the instance of `Dictionary`.
        TODO: docstr
        max_num_text (int): Maximum number of text instances. Defaults to 60.
        loss (dict, optional): Config to build loss. Defaults to None.
        postprocessor (dict, optional): Config to build postprocessor.
            Defaults to None.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 dictionary: Union[Dict, Dictionary],
                 num_bins: int = 1000,
                 n_head: int = 8,
                 d_model: int = 256,
                 d_feedforward: int = 1024,
                 normalize_before: bool = True,
                 dropout: float = 0.1,
                 max_num_text: int = 60,
                 module_loss: Optional[Dict] = None,
                 postprocessor: Optional[Dict] = None,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:

        # TODO: fix hardcode
        self.max_seq_len = (2 + 25) * max_num_text + 1
        super().__init__(
            dictionary=dictionary,
            module_loss=module_loss,
            postprocessor=postprocessor,
            max_seq_len=self.max_seq_len,
            init_cfg=init_cfg)
        self.num_bins = num_bins
        self.shifted_seq_end_idx = self.num_bins + self.dictionary.seq_end_idx
        self.shifted_start_idx = self.num_bins + self.dictionary.start_idx

        actual_num_classes = self.dictionary.num_classes + num_bins

        self.embedding = DecoderEmbeddings(
            actual_num_classes, self.dictionary.padding_idx + num_bins,
            d_model, self.max_seq_len, dropout)
        self.pos_embedding = PositionEmbeddingSine(d_model // 2)

        self.vocab_embed = self._gen_vocab_embed(d_model, d_model,
                                                 actual_num_classes, 3)
        encoder_layer = TransformerEncoderLayer(d_model, n_head, d_feedforward,
                                                dropout, 'relu',
                                                normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        num_encoder_layers = 6
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers,
                                          encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, n_head, d_feedforward,
                                                dropout, 'relu',
                                                normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        num_decoder_layers = 6
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=False)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _gen_vocab_embed(self, input_dim: int, hidden_dim: int,
                         output_dim: int, num_layers: int) -> nn.Module:
        """Generate vocab embedding layer."""
        net = nn.Sequential()
        h = [hidden_dim] * (num_layers - 1)
        for i, (n, k) in enumerate(zip([input_dim] + h, h + [output_dim])):
            net.add_module(f'layer-{i}', nn.Linear(n, k))
            if i < num_layers - 1:
                net.add_module(f'relu-{i}', nn.ReLU())
        return net

    def forward_train(
        self,
        feat: Optional[torch.Tensor] = None,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextSpottingDataSample]] = None
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
        mask, pos_embed, memory, query_embed = self._embed(
            out_enc, data_samples)

        padded_targets = [
            data_sample.gt_instances.padded_indexes
            for data_sample in data_samples
        ]
        padded_targets = torch.stack(padded_targets, dim=0).to(out_enc.device)
        # we don't need eos here
        tgt = self.embedding(padded_targets[:, :-1]).permute(1, 0, 2)
        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed[:len(tgt)],
            tgt_mask=self._generate_square_subsequent_mask(len(tgt)).to(
                tgt.device))
        return self.vocab_embed(hs[-1].transpose(0, 1))

    def forward_test(
        self,
        feat: Optional[torch.Tensor] = None,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextSpottingDataSample]] = None
    ) -> torch.Tensor:
        """Forward for testing.

        Args:
            feat (torch.Tensor, optional): The feature map from backbone of
                shape :math:`(N, E, H, W)`. Defaults to None.
            out_enc (torch.Tensor, optional): Encoder output. Defaults to None.
            data_samples (Sequence[TextRecogDataSample]): Batch of
                TextRecogDataSample, containing gt_text information. Defaults
                to None.
        """

        batch_size = out_enc.shape[0]
        mask, pos_embed, memory, query_embed = self._embed(
            out_enc, data_samples)

        max_probs = []
        seq = torch.zeros(
            batch_size, 1, dtype=torch.long).to(
                out_enc.device) + self.shifted_start_idx
        for i in range(self.max_seq_len):
            tgt = self.embedding(seq).permute(1, 0, 2)
            hs = self.decoder(
                tgt,
                memory,
                memory_key_padding_mask=mask,
                pos=pos_embed,
                query_pos=query_embed[:len(tgt)],
                tgt_mask=self._generate_square_subsequent_mask(len(tgt)).to(
                    tgt.device))  # bs, 1, E ?
            out = self.vocab_embed(hs.transpose(1, 2)[-1, :, -1, :])
            out = out.softmax(-1)

            # bins chars unk eos seq_eos sos padding
            if i % 27 == 0:  # coordinate or eos
                out[:, self.num_bins:self.shifted_seq_end_idx] = 0
                out[:, self.shifted_seq_end_idx + 1:] = 0
            elif i % 27 == 1:  # coordinate
                out[:, self.num_bins:] = 0
            else:  # chars
                out[:, :self.num_bins] = 0
                out[:, self.shifted_seq_end_idx:] = 0

            max_prob, extra_seq = torch.max(out, dim=-1, keepdim=True)
            # prob, extra_seq = out.topk(dim=-1, k=1)
            # work for single batch only (original implementation)
            # TODO: optimize for multi-batch
            seq = torch.cat([seq, extra_seq], dim=-1)
            max_probs.append(max_prob)
            if extra_seq[0] == self.shifted_seq_end_idx:
                break

        max_probs = torch.cat(max_probs, dim=-1)
        max_probs = max_probs[:, :-1]  # remove seq_eos
        seq = seq[:, 1:-1]  # remove start index and seq_eos
        return max_probs, seq

    def _embed(self, out_enc, data_samples):
        bs, c, h, w = out_enc.shape
        mask, pos_embed = self._gen_mask(out_enc, data_samples)
        out_enc = out_enc.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        # TODO move encoder to mmcv
        memory = self.encoder(
            out_enc, src_key_padding_mask=mask, pos=pos_embed.half())

        query_embed = self.embedding.position_embeddings.weight.unsqueeze(1)
        query_embed = query_embed.repeat(1, bs, 1)
        return mask, pos_embed, memory, query_embed

    def _generate_square_subsequent_mask(self, size):
        r"""Generate a square mask for the sequence. The masked positions are
            filled with float('-inf'). Unmasked positions are filled with
            float(0.0).
        """
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def _gen_mask(self, out_enc, data_samples):
        bs, _, h, w = out_enc.shape
        masks = torch.ones((bs, h, w), dtype=bool, device=out_enc.device)
        for i, data_sample in enumerate(data_samples):
            img_h, img_w = data_sample.img_shape
            masks[i, :img_h, :img_w] = False
        masks = F.interpolate(
            masks[None].float(), size=(h, w)).to(torch.bool)[0]
        return masks, self.pos_embedding(masks)


class DecoderEmbeddings(nn.Module):

    def __init__(self, num_classes: int, padding_idx: int, hidden_dim,
                 max_position_embeddings, dropout):
        super(DecoderEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            num_classes, hidden_dim, padding_idx=padding_idx)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_dim)

        self.LayerNorm = torch.nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        input_shape = x.size()
        seq_length = input_shape[1]
        device = x.device

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_embeds = self.word_embeddings(x)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = input_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self,
                 decoder_layer,
                 num_layers,
                 norm=None,
                 return_intermediate=False):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self,
                tgt,
                memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos)

        if self.norm is not None:
            # nn.LayerNorm(d_model)
            output = self.norm(output)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 normalize_before=False):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q,
            k,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self,
                    src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q,
            k,
            value=src2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 normalize_before=False):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     tgt,
                     memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self,
                    tgt,
                    memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self,
                tgt,
                memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask,
                                 pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string."""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(F'activation should be relu/gelu, not {activation}.')
