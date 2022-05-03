# Copyright (c) OpenMMLab. All rights reserved.
from queue import PriorityQueue

import torch
import torch.nn.functional as F

import mmocr.utils as utils
from mmocr.models.builder import DECODERS
from . import ASTERDecoder


class DecodeNodes:
    """Node class to save decoded char indices and scores.

    Args:
        indexes (list[int]): Char indices that decoded yes.
        scores (list[float]): Char scores that decoded yes.
    """

    def __init__(self, state, indexes=[1], scores=[0.9]):
        assert utils.is_type_list(indexes, int)
        assert utils.is_type_list(scores, float)
        assert utils.equal_len(indexes, scores)

        self.indexes = indexes
        self.scores = scores
        self.state = [state]

    def eval(self):
        """Calculate accumulated score."""
        accu_score = sum(self.scores)
        return accu_score


@DECODERS.register_module()
class ASTERDecoderWithBs(ASTERDecoder):
    """Aster Decoder module with beam-search .

    Args:
        beam_width (int): Width for beam search.
    """

    def __init__(self,
                 in_channels,
                 num_classes,
                 s_Dim,
                 Atten_Dim,
                 max_seq_len=40,
                 beam_width=5,
                 start_idx=0,
                 padding_idx=92,
                 init_cfg=None,
                 **kwargs):

        super().__init__(
            in_channels,
            num_classes,
            s_Dim,
            Atten_Dim,
            max_seq_len,
            beam_width,
            start_idx,
            init_cfg=init_cfg)

        assert isinstance(beam_width, int)
        assert beam_width > 0
        self.beam_width = beam_width
        self.start_idx = start_idx

    def forward_test(self, feat, out_enc, img_metas):
        x = out_enc
        batch_size = x.size(0)
        assert batch_size == 1, 'batch size must bu 1 for beam search.'
        state = torch.zeros(1, batch_size, self.s_Dim).to(feat.device)
        y_prev = torch.zeros(
            (batch_size)).fill_(self.num_classes).to(feat.device)

        nodes = PriorityQueue()
        init_node = DecodeNodes(state, [self.start_idx], [0.0])
        nodes.put((-init_node.eval(), init_node))
        for i in range(1, self.max_seq_len + 1):
            next_nodes = []
            beam_width = self.beam_width if i > 1 else 1
            for beam_w in range(beam_width):
                _, node = nodes.get()
                if i > 1:
                    y_prev = torch.tensor(node.indexes[i - 1]).view(1).to(
                        feat.device)
                    state = node.state[0][1]
                decoder_output, temp_state = self.decoder(x, state, y_prev)
                decoder_output = F.softmax(decoder_output, dim=1)

                topk_score, topk_idx = torch.topk(decoder_output,
                                                  self.beam_width)
                topk_score, topk_idx = topk_score.squeeze(0), topk_idx.squeeze(
                    0)

                for k in range(self.beam_width):
                    kth_score = topk_score[k].item()
                    kth_idx = topk_idx[k].item()
                    next_node = DecodeNodes(node.state + [temp_state],
                                            node.indexes + [kth_idx],
                                            node.scores + [kth_score])
                    delta = k * 1e-6
                    next_nodes.append(
                        ((-node.eval() - kth_score - delta), next_node))
            while not nodes.empty():
                nodes.get()

            for next_node in next_nodes:
                nodes.put(next_node)

        best_node = nodes.get()
        outputs = torch.zeros(batch_size, self.max_seq_len, self.num_classes)

        for i in range(self.max_seq_len):
            idx = best_node[1].indexes[i + 1]
            outputs[0, i, idx] = best_node[1].scores[i + 1]

        return outputs[:, 1:, :]
