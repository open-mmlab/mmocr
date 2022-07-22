# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
from mmdet.models.losses import accuracy
from torch import Tensor, nn

from mmocr.data import KIEDataSample
from mmocr.registry import MODELS


@MODELS.register_module()
class SDMGRModuleLoss(nn.Module):
    """The implementation the loss of key information extraction proposed in
    the paper: `Spatial Dual-Modality Graph Reasoning for Key Information
    Extraction <https://arxiv.org/abs/2103.14470>`_.

    Args:
        weight_node (float): Weight of node loss. Defaults to 1.0.
        weight_edge (float): Weight of edge loss. Defaults to 1.0.
        ignore_idx (int): Node label to ignore. Defaults to -100.
    """

    def __init__(self,
                 weight_node: float = 1.0,
                 weight_edge: float = 1.0,
                 ignore_idx: int = -100) -> None:
        super().__init__()
        # TODO: Use MODELS.build after DRRG loss has been merged
        self.loss_node = nn.CrossEntropyLoss(ignore_index=ignore_idx)
        self.loss_edge = nn.CrossEntropyLoss(ignore_index=-1)
        self.weight_node = weight_node
        self.weight_edge = weight_edge
        self.ignore_idx = ignore_idx

    def forward(self, preds: Tuple[Tensor, Tensor],
                data_samples: List[KIEDataSample]) -> Dict:
        """Forward function.

        Args:
            preds (tuple(Tensor, Tensor)):
            data_samples (list[KIEDataSample]): A list of datasamples
                containing ``gt_instances.labels`` and
                ``gt_instances.edge_labels``.

        Returns:
            dict(str, Tensor): Loss dict, containing ``loss_node``,
            ``loss_edge``, ``acc_node`` and ``acc_edge``.
        """
        node_preds, edge_preds = preds
        node_gts, edge_gts = [], []
        for data_sample in data_samples:
            node_gts.append(data_sample.gt_instances.labels)
            edge_gts.append(data_sample.gt_instances.edge_labels.reshape(-1))
        node_gts = torch.cat(node_gts).long()
        edge_gts = torch.cat(edge_gts).long()

        node_valids = torch.nonzero(
            node_gts != self.ignore_idx, as_tuple=False).reshape(-1)
        edge_valids = torch.nonzero(edge_gts != -1, as_tuple=False).reshape(-1)
        return dict(
            loss_node=self.weight_node * self.loss_node(node_preds, node_gts),
            loss_edge=self.weight_edge * self.loss_edge(edge_preds, edge_gts),
            acc_node=accuracy(node_preds[node_valids], node_gts[node_valids]),
            acc_edge=accuracy(edge_preds[edge_valids], edge_gts[edge_valids]))
