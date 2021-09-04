# Copyright (c) OpenMMLab. All rights reserved.
from queue import PriorityQueue

import torch
import torch.nn.functional as F

import mmocr.utils as utils
from mmocr.models.builder import DECODERS
from . import ParallelSARDecoder


class DecodeNode:
    """Node class to save decoded char indices and scores.

    Args:
        indexes (list[int]): Char indices that decoded yes.
        scores (list[float]): Char scores that decoded yes.
    """

    def __init__(self, indexes=[1], scores=[0.9]):
        assert utils.is_type_list(indexes, int)
        assert utils.is_type_list(scores, float)
        assert utils.equal_len(indexes, scores)

        self.indexes = indexes
        self.scores = scores

    def eval(self):
        """Calculate accumulated score."""
        accu_score = sum(self.scores)
        return accu_score


@DECODERS.register_module()
class ParallelSARDecoderWithBS(ParallelSARDecoder):
    """Parallel Decoder module with beam-search in SAR.

    Args:
        beam_width (int): Width for beam search.
    """

    def __init__(self,
                 beam_width=5,
                 num_classes=37,
                 enc_bi_rnn=False,
                 dec_bi_rnn=False,
                 dec_do_rnn=0,
                 dec_gru=False,
                 d_model=512,
                 d_enc=512,
                 d_k=64,
                 pred_dropout=0.0,
                 max_seq_len=40,
                 mask=True,
                 start_idx=0,
                 padding_idx=0,
                 pred_concat=False,
                 init_cfg=None,
                 **kwargs):
        super().__init__(
            num_classes,
            enc_bi_rnn,
            dec_bi_rnn,
            dec_do_rnn,
            dec_gru,
            d_model,
            d_enc,
            d_k,
            pred_dropout,
            max_seq_len,
            mask,
            start_idx,
            padding_idx,
            pred_concat,
            init_cfg=init_cfg)
        assert isinstance(beam_width, int)
        assert beam_width > 0

        self.beam_width = beam_width

    def forward_test(self, feat, out_enc, img_metas):
        assert utils.is_type_list(img_metas, dict)
        assert len(img_metas) == feat.size(0)

        valid_ratios = [
            img_meta.get('valid_ratio', 1.0) for img_meta in img_metas
        ] if self.mask else None

        seq_len = self.max_seq_len
        bsz = feat.size(0)
        assert bsz == 1, 'batch size must be 1 for beam search.'

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

        # Initialize beam-search queue
        q = PriorityQueue()
        init_node = DecodeNode([self.start_idx], [0.0])
        q.put((-init_node.eval(), init_node))

        for i in range(1, seq_len + 1):
            next_nodes = []
            beam_width = self.beam_width if i > 1 else 1
            for _ in range(beam_width):
                _, node = q.get()

                input_seq = torch.clone(decoder_input)  # bsz * T * emb_dim
                # fill previous input tokens (step 1...i) in input_seq
                for t, index in enumerate(node.indexes):
                    input_token = torch.full((bsz, ),
                                             index,
                                             device=input_seq.device,
                                             dtype=torch.long)
                    input_token = self.embedding(input_token)  # bsz * emb_dim
                    input_seq[:, t + 1, :] = input_token

                output_seq = self._2d_attention(
                    input_seq, feat, out_enc, valid_ratios=valid_ratios)

                output_char = output_seq[:, i, :]  # bsz * num_classes
                output_char = F.softmax(output_char, -1)
                topk_value, topk_idx = output_char.topk(self.beam_width, dim=1)
                topk_value, topk_idx = topk_value.squeeze(0), topk_idx.squeeze(
                    0)

                for k in range(self.beam_width):
                    kth_score = topk_value[k].item()
                    kth_idx = topk_idx[k].item()
                    next_node = DecodeNode(node.indexes + [kth_idx],
                                           node.scores + [kth_score])
                    delta = k * 1e-6
                    next_nodes.append(
                        (-node.eval() - kth_score - delta, next_node))
                    # Use minus since priority queue sort
                    # with ascending order

            while not q.empty():
                q.get()

            # Put all candidates to queue
            for next_node in next_nodes:
                q.put(next_node)

        best_node = q.get()
        num_classes = self.num_classes - 1  # ignore padding index
        outputs = torch.zeros(bsz, seq_len, num_classes)
        for i in range(seq_len):
            idx = best_node[1].indexes[i + 1]
            outputs[0, i, idx] = best_node[1].scores[i + 1]

        return outputs
