# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmengine.model import BaseModule
from torch import Tensor, nn
from torch.nn import functional as F

from mmocr.data import KIEDataSample
from mmocr.models.textrecog.dictionary import Dictionary
from mmocr.registry import MODELS, TASK_UTILS


@MODELS.register_module()
class SDMGRHead(BaseModule):
    """SDMGR Head.

    Args:
        dictionary (dict or :obj:`Dictionary`): The config for `Dictionary` or
            the instance of `Dictionary`.
        num_classes (int): Number of class labels. Defaults to 26.
        visual_dim (int): Dimension of visual features :math:`E`. Defaults to
            64.
        fusion_dim (int): Dimension of fusion layer. Defaults to 1024.
        node_input (int): Dimension of raw node embedding. Defaults to 32.
        node_embed (int): Dimension of node embedding. Defaults to 256.
        edge_input (int): Dimension of raw edge embedding. Defaults to 5.
        edge_embed (int): Dimension of edge embedding. Defaults to 256.
        num_gnn (int): Number of GNN layers. Defaults to 2.
        bidirectional (bool): Whether to use bidirectional RNN to embed nodes.
            Defaults to False.
        relation_norm (float): Norm to map value from one range to another.=
            Defaults to 10.
        module_loss (dict): Module Loss config. Defaults to
            ``dict(type='SDMGRModuleLoss')``.
        postprocessor (dict): Postprocessor config. Defaults to
            ``dict(type='SDMGRPostProcessor')``.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(
        self,
        dictionary: Union[Dictionary, Dict],
        num_classes: int = 26,
        visual_dim: int = 64,
        fusion_dim: int = 1024,
        node_input: int = 32,
        node_embed: int = 256,
        edge_input: int = 5,
        edge_embed: int = 256,
        num_gnn: int = 2,
        bidirectional: bool = False,
        relation_norm: float = 10.,
        module_loss: Dict = dict(type='SDMGRModuleLoss'),
        postprocessor: Dict = dict(type='SDMGRPostProcessor'),
        init_cfg: Optional[Union[Dict, List[Dict]]] = dict(
            type='Normal', override=dict(name='edge_embed'), mean=0, std=0.01)
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(dictionary, (dict, Dictionary))
        if isinstance(dictionary, dict):
            self.dictionary = TASK_UTILS.build(dictionary)
        elif isinstance(dictionary, Dictionary):
            self.dictionary = dictionary

        self.fusion = FusionBlock([visual_dim, node_embed], node_embed,
                                  fusion_dim)
        self.node_embed = nn.Embedding(self.dictionary.num_classes, node_input,
                                       self.dictionary.padding_idx)
        hidden = node_embed // 2 if bidirectional else node_embed
        self.rnn = nn.LSTM(
            input_size=node_input,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional)
        self.edge_embed = nn.Linear(edge_input, edge_embed)
        self.gnn_layers = nn.ModuleList(
            [GNNLayer(node_embed, edge_embed) for _ in range(num_gnn)])
        self.node_cls = nn.Linear(node_embed, num_classes)
        self.edge_cls = nn.Linear(edge_embed, 2)
        self.module_loss = MODELS.build(module_loss)
        self.postprocessor = MODELS.build(postprocessor)
        self.relation_norm = relation_norm

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: List[KIEDataSample]) -> Dict:
        """Calculate losses from a batch of inputs and data samples.
        Args:
            batch_inputs (torch.Tensor): Shape :math:`(N, E)`.
            batch_data_samples (List[KIEDataSample]): List of data samples.

        Returns:
            dict[str, tensor]: A dictionary of loss components.
        """
        preds = self.forward(batch_inputs, batch_data_samples)
        return self.module_loss(preds, batch_data_samples)

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: List[KIEDataSample]
                ) -> List[KIEDataSample]:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (torch.Tensor): Shape :math:`(N, E)`.
            batch_data_samples (List[KIEDataSample]): List of data samples.

        Returns:
            List[KIEDataSample]: A list of datasamples of prediction results.
            Results are stored in ``pred_instances.labels``,
            ``pred_instances.scores``, ``pred_instances.edge_labels`` and
            ``pred_instances.edge_scores``.

            - labels (Tensor): An integer tensor of shape (N, ) indicating bbox
              labels for each image.
            - scores (Tensor): A float tensor of shape (N, ), indicating the
              confidence scores for node label predictions.
            - edge_labels (Tensor): An integer tensor of shape (N, N)
              indicating the connection between nodes. Options are 0, 1.
            - edge_scores (Tensor): A float tensor of shape (N, ), indicating
              the confidence scores for edge predictions.
        """
        preds = self.forward(batch_inputs, batch_data_samples)
        return self.postprocessor(preds, batch_data_samples)

    def forward(self, batch_inputs: Tensor,
                batch_data_samples: List[KIEDataSample]
                ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch_inputs (torch.Tensor): Shape :math:`(N, E)`.
            batch_data_samples (List[KIEDataSample]): List of data samples.

        Returns:
            tuple(Tensor, Tensor):

            - node_cls (Tensor): Raw logits scores for nodes. Shape
              :math:`(N, C_{l})` where :math:`C_{l}` is number of classes.
            - edge_cls (Tensor): Raw logits scores for edges. Shape
              :math:`(N * N, 2)`.
        """

        device = self.node_embed.weight.device

        node_nums, char_nums, all_nodes = self.convert_texts(
            batch_data_samples)

        embed_nodes = self.node_embed(all_nodes.to(device).long())
        rnn_nodes, _ = self.rnn(embed_nodes)

        nodes = rnn_nodes.new_zeros(*rnn_nodes.shape[::2])
        all_nums = torch.cat(char_nums).to(device)
        valid = all_nums > 0
        nodes[valid] = rnn_nodes[valid].gather(
            1, (all_nums[valid] - 1).unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, rnn_nodes.size(-1))).squeeze(1)

        if batch_inputs is not None:
            nodes = self.fusion([batch_inputs, nodes])

        relations = self.compute_relations(batch_data_samples)
        all_edges = torch.cat(
            [relation.view(-1, relation.size(-1)) for relation in relations],
            dim=0)
        embed_edges = self.edge_embed(all_edges.float())
        embed_edges = F.normalize(embed_edges)

        for gnn_layer in self.gnn_layers:
            nodes, embed_edges = gnn_layer(nodes, embed_edges, node_nums)

        node_cls, edge_cls = self.node_cls(nodes), self.edge_cls(embed_edges)
        return node_cls, edge_cls

    def convert_texts(
        self, data_samples: List[KIEDataSample]
    ) -> Tuple[List[Tensor], List[Tensor], Tensor]:
        """Extract texts in datasamples and pack them into a batch.

        Args:
            data_samples (List[KIEDataSample]): List of data samples.

        Returns:
            tuple(List[int], List[Tensor], Tensor):

            - node_nums (List[int]): A list of node numbers for each
              sample.
            - char_nums (List[Tensor]): A list of character numbers for each
              sample.
            - nodes (Tensor): A tensor of shape :math:`(N, C)` where
              :math:`C` is the maximum number of characters in a sample.
        """
        node_nums, char_nums = [], []
        max_len = -1
        text_idxs = []
        for data_sample in data_samples:
            node_nums.append(len(data_sample.gt_instances.texts))
            for text in data_sample.gt_instances.texts:
                text_idxs.append(self.dictionary.str2idx(text))
                max_len = max(max_len, len(text))

        nodes = torch.zeros((sum(node_nums), max_len),
                            dtype=torch.long) + self.dictionary.padding_idx
        for i, text_idx in enumerate(text_idxs):
            nodes[i, :len(text_idx)] = torch.LongTensor(text_idx)
        char_nums = (nodes != self.dictionary.padding_idx).sum(-1).split(
            node_nums, dim=0)
        return node_nums, char_nums, nodes

    def compute_relations(self, data_samples: List[KIEDataSample]) -> Tensor:
        """Compute the relations between every two boxes for each datasample,
        then return the concatenated relations."""

        relations = []
        for data_sample in data_samples:
            bboxes = data_sample.gt_instances.bboxes
            x1, y1 = bboxes[:, 0:1], bboxes[:, 1:2]
            x2, y2 = bboxes[:, 2:3], bboxes[:, 3:4]
            w, h = torch.clamp(
                x2 - x1 + 1, min=1), torch.clamp(
                    y2 - y1 + 1, min=1)
            dx = (x1.t() - x1) / self.relation_norm
            dy = (y1.t() - y1) / self.relation_norm
            xhh, xwh = h.T / h, w.T / h
            whs = w / h + torch.zeros_like(xhh)
            relation = torch.stack([dx, dy, whs, xhh, xwh], -1).float()
            relations.append(relation)
        return relations


class GNNLayer(nn.Module):
    """GNN layer for SDMGR.

    Args:
        node_dim (int): Dimension of node embedding. Defaults to 256.
        edge_dim (int): Dimension of edge embedding. Defaults to 256.
    """

    def __init__(self, node_dim: int = 256, edge_dim: int = 256) -> None:
        super().__init__()
        self.in_fc = nn.Linear(node_dim * 2 + edge_dim, node_dim)
        self.coef_fc = nn.Linear(node_dim, 1)
        self.out_fc = nn.Linear(node_dim, node_dim)
        self.relu = nn.ReLU()

    def forward(self, nodes: Tensor, edges: Tensor,
                nums: List[int]) -> Tuple[Tensor, Tensor]:
        """Forward function.

        Args:
            nodes (Tensor): Concatenated node embeddings.
            edges (Tensor): Concatenated edge embeddings.
            nums (List[int]): List of number of nodes in each batch.

        Returns:
            tuple(Tensor, Tensor):

            - nodes (Tensor): New node embeddings.
            - edges (Tensor): New edge embeddings.
        """
        start, cat_nodes = 0, []
        for num in nums:
            sample_nodes = nodes[start:start + num]
            cat_nodes.append(
                torch.cat([
                    sample_nodes.unsqueeze(1).expand(-1, num, -1),
                    sample_nodes.unsqueeze(0).expand(num, -1, -1)
                ], -1).view(num**2, -1))
            start += num
        cat_nodes = torch.cat([torch.cat(cat_nodes), edges], -1)
        cat_nodes = self.relu(self.in_fc(cat_nodes))
        coefs = self.coef_fc(cat_nodes)

        start, residuals = 0, []
        for num in nums:
            residual = F.softmax(
                -torch.eye(num).to(coefs.device).unsqueeze(-1) * 1e9 +
                coefs[start:start + num**2].view(num, num, -1), 1)
            residuals.append(
                (residual *
                 cat_nodes[start:start + num**2].view(num, num, -1)).sum(1))
            start += num**2

        nodes += self.relu(self.out_fc(torch.cat(residuals)))
        return nodes, cat_nodes


class FusionBlock(nn.Module):
    """Fusion block of SDMGR.

    Args:
        input_dims (tuple(int, int)): Visual dimension and node embedding
            dimension.
        output_dim (int): Output dimension.
        mm_dim (int): Model dimension. Defaults to 1600.
        chunks (int): Number of chunks. Defaults to 20.
        rank (int): Rank number. Defaults to 15.
        shared (bool): Whether to share the project layer between visual and
            node embedding features. Defaults to False.
        dropout_input (float): Dropout rate after the first projection layer.
            Defaults to 0.
        dropout_pre_lin (float): Dropout rate before the final project layer.
            Defaults to 0.
        dropout_pre_lin (float): Dropout rate after the final project layer.
            Defaults to 0.
        pos_norm (str): The normalization position. Options are 'before_cat'
            and 'after_cat'. Defaults to 'before_cat'.
    """

    def __init__(self,
                 input_dims: Tuple[int, int],
                 output_dim: int,
                 mm_dim: int = 1600,
                 chunks: int = 20,
                 rank: int = 15,
                 shared: bool = False,
                 dropout_input: float = 0.,
                 dropout_pre_lin: float = 0.,
                 dropout_output: float = 0.,
                 pos_norm: str = 'before_cat') -> None:
        super().__init__()
        self.rank = rank
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        assert (pos_norm in ['before_cat', 'after_cat'])
        self.pos_norm = pos_norm
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        self.linear1 = (
            self.linear0 if shared else nn.Linear(input_dims[1], mm_dim))
        self.merge_linears0 = nn.ModuleList()
        self.merge_linears1 = nn.ModuleList()
        self.chunks = self.chunk_sizes(mm_dim, chunks)
        for size in self.chunks:
            ml0 = nn.Linear(size, size * rank)
            self.merge_linears0.append(ml0)
            ml1 = ml0 if shared else nn.Linear(size, size * rank)
            self.merge_linears1.append(ml1)
        self.linear_out = nn.Linear(mm_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        bs = x1.size(0)
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        x0_chunks = torch.split(x0, self.chunks, -1)
        x1_chunks = torch.split(x1, self.chunks, -1)
        zs = []
        for x0_c, x1_c, m0, m1 in zip(x0_chunks, x1_chunks,
                                      self.merge_linears0,
                                      self.merge_linears1):
            m = m0(x0_c) * m1(x1_c)  # bs x split_size*rank
            m = m.view(bs, self.rank, -1)
            z = torch.sum(m, 1)
            if self.pos_norm == 'before_cat':
                z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
                z = F.normalize(z)
            zs.append(z)
        z = torch.cat(zs, 1)
        if self.pos_norm == 'after_cat':
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z)

        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z

    @staticmethod
    def chunk_sizes(dim: int, chunks: int) -> List[int]:
        """Compute chunk sizes."""
        split_size = (dim + chunks - 1) // chunks
        sizes_list = [split_size] * chunks
        sizes_list[-1] = sizes_list[-1] - (sum(sizes_list) - dim)
        return sizes_list
