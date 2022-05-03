# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLSTM(nn.Module):

    def __init__(self, H_Dim, S_Dim, Atten_Dim):
        super().__init__()
        """
        Args:
            S_Dim:  The state of dimensions:512
            H_Dim:  The output of encoder of dimensions:512
            Atten_Dim: Then attention layer of dimensions:512
        """
        self.H_Dim = H_Dim
        self.S_Dim = S_Dim
        self.Atten_Dim = Atten_Dim

        self.V_h = nn.Linear(self.H_Dim, self.Atten_Dim)
        self.W_s = nn.Linear(self.S_Dim, self.Atten_Dim)
        self.W_e = nn.Linear(self.Atten_Dim, 1)

    def forward(self, H_t, S_t):
        """

        Args:
            h_t(Tensor): A Tensor of shape(N, W, C)
            s_t1(Tensor): is the state at t-1 moment, shape(1, b, hidden_dim)
        Returns:
            alpha(Tensor): attentional weights at t moment
        """
        b, T, _ = H_t.size()
        H_t = H_t.view(-1, self.H_Dim)
        H_a = self.V_h(H_t)
        H_a = H_a.view(b, T, -1)

        S_t = S_t.squeeze(0)
        s_Proj = self.W_s(S_t)
        s_Proj = torch.unsqueeze(s_Proj, 1)
        s_Proj = s_Proj.expand(b, T, self.Atten_Dim)

        WHtanh = torch.tanh(s_Proj + H_a)
        WHtanh = WHtanh.view(-1, self.Atten_Dim)
        e_t = self.W_e(WHtanh)

        e_t = e_t.view(b, T)
        alpha = F.softmax(e_t, dim=1)
        return alpha


class Decoder(nn.Module):

    def __init__(
        self,
        h_Dim,
        y_Dim,
        s_Dim,
        Atten_Dim,
    ):
        self.h_Dim = h_Dim
        self.y_Dim = y_Dim
        self.s_Dim = s_Dim
        self.Atten_Dim = Atten_Dim
        self.emb_Dim = Atten_Dim
        super().__init__()
        self.Attention = AttentionLSTM(h_Dim, s_Dim, Atten_Dim)
        self.y_emd = nn.Embedding(y_Dim + 1, self.emb_Dim)
        self.gru = nn.GRU(
            input_size=h_Dim + self.emb_Dim,
            hidden_size=s_Dim,
            batch_first=True)
        self.embeding = nn.Linear(s_Dim, y_Dim)

    def forward(self, h_i, s_t, y_t):
        """
        Args:
            h_i:  is the output of BLSTM
            s_t:  is the state at t-1 moment
            y_t:  is the y at t-1 moment

        Returns:
            output: Xt is taken for predicting the current-setp symbol
            state: St is new state by recurrent unit
        """
        b, T, _ = h_i.size()
        alpha = self.Attention(h_i, s_t)
        g_t = torch.bmm(alpha.unsqueeze(1), h_i).squeeze(1)
        y_one = self.y_emd(y_t.long())
        x_t, state = self.gru(torch.cat([y_one, g_t], 1).unsqueeze(1), s_t)
        output = x_t.squeeze(1)

        output = self.embeding(output)
        #
        # outputs = output.unsqueeze(1)

        return output, state
