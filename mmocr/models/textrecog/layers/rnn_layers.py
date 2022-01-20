# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super().__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class AttentionGRUCell(nn.Module):
    """GRU cell with attention on encoder outputs, usually designed for
    decoder.

    Args:
        num_channels (int): Number of channels of hidden vectors :math:`E`.
        out_channels (int): Number of channels of output vector :math:`C`.
        dropout (float): Dropout rate for the embedding vector.
    """

    def __init__(self, num_channels, out_channels, dropout=0.1):
        super().__init__()
        self.hidden_size = num_channels
        # self.output_size = cfg.MODEL.BATEXT.VOC_SIZE + 1
        self.output_size = out_channels

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.vat = nn.Linear(self.hidden_size, 1)

    def forward(self, x, hidden, encoder_outputs):
        '''
        Args:
            x (Tensor): Shape :math:`(N)` or :math:`(N, 1)`.
            hidden (Tensor): Shape :math:`(1, N, E)`.
            encoder_outputs (Tensor): Shape :math:`(T, N, E)`.

        Returns:
            tuple(Tensor): A tuple of three tensors: the output at current step
            :math:`(N, C)`, the hidden state :math:`(1, N, E)` and the
            attention weights :math:`(N, T)`.
        '''
        x = x.squeeze()
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)  # (N, E)

        N = encoder_outputs.shape[1]

        alpha = hidden + encoder_outputs
        alpha = alpha.reshape(-1, alpha.shape[-1])  # (T * N, E)
        attn_weights = self.vat(torch.tanh(alpha))  # (T * N, 1)
        attn_weights = attn_weights.view(-1, 1, N).permute(
            (2, 1, 0))  # (T, 1, N)  -> (N, 1, T)
        attn_weights = F.softmax(attn_weights, dim=2)

        attn_applied = torch.matmul(attn_weights,
                                    encoder_outputs.permute(
                                        (1, 0, 2)))  # (N, T, E)

        if embedded.dim() == 1:
            embedded = embedded.unsqueeze(0)
        output = torch.cat((embedded, attn_applied.squeeze(1)), 1)  # (N, 2E)
        output = self.attn_combine(output).unsqueeze(0)  # (1, N, E)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)  # (1, N, E)

        output = F.log_softmax(self.out(output[0]), dim=1)  # (N, C)
        return output, hidden, attn_weights

    def init_hidden(self, batch_size):
        """Returns a zero hidden vector of shape :math:`(1, N, E)` where N is
        ``batch_size``."""
        result = torch.zeros(1, batch_size, self.hidden_size)
        return result
