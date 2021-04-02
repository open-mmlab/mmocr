import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class MeanAggregator(nn.Module):

    def forward(self, features, A):
        x = torch.bmm(A, features)
        return x


class GraphConv(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.FloatTensor(in_dim * 2, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        init.xavier_uniform_(self.weight)
        init.constant_(self.bias, 0)
        self.agg = MeanAggregator()

    def forward(self, features, A):
        b, n, d = features.shape
        assert d == self.in_dim
        agg_feats = self.agg(features, A)
        cat_feats = torch.cat([features, agg_feats], dim=2)
        out = torch.einsum('bnd,df->bnf', (cat_feats, self.weight))
        out = F.relu(out + self.bias)
        return out


class GCN(nn.Module):
    """Predict linkage between instances. This was from repo
    https://github.com/Zhongdao/gcn_clustering: Linkage Based Face Clustering
    via Graph Convolution Network.

    [https://arxiv.org/abs/1903.11306]

    Args:
        in_dim(int): The input dimension.
        out_dim(int): The output dimension.
    """

    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()
        self.bn0 = nn.BatchNorm1d(in_dim, affine=False).float()
        self.conv1 = GraphConv(in_dim, 512)
        self.conv2 = GraphConv(512, 256)
        self.conv3 = GraphConv(256, 128)
        self.conv4 = GraphConv(128, 64)

        self.classifier = nn.Sequential(
            nn.Linear(64, out_dim), nn.PReLU(out_dim), nn.Linear(out_dim, 2))

    def forward(self, x, A, one_hop_indexes, train=True):

        B, N, D = x.shape

        x = x.view(-1, D)
        x = self.bn0(x)
        x = x.view(B, N, D)

        x = self.conv1(x, A)
        x = self.conv2(x, A)
        x = self.conv3(x, A)
        x = self.conv4(x, A)
        k1 = one_hop_indexes.size(-1)
        dout = x.size(-1)
        edge_feat = torch.zeros(B, k1, dout)
        for b in range(B):
            edge_feat[b, :, :] = x[b, one_hop_indexes[b]]
        edge_feat = edge_feat.view(-1, dout).to(x.device)
        pred = self.classifier(edge_feat)

        # shape: (B*k1)x2
        return pred
