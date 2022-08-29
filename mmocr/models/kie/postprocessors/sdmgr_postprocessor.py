# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import numpy as np
import torch
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmocr.registry import MODELS
from mmocr.structures import KIEDataSample


@MODELS.register_module()
class SDMGRPostProcessor:
    """Postprocessor for SDMGR. It converts the node and edge scores into
    labels and edge labels. If the link_type is not "none", it reconstructs the
    edge labels with different strategies specified by ``link_type``, which is
    generally known as the "openset" mode. In "openset" mode, only the edges
    connecting from "key" to "value" nodes will be constructed.

    Args:
        link_type (str): The type of link to be constructed.
            Defaults to 'none'. Options are:

            - 'none': The simplest link type involving no edge
              postprocessing. The edge prediction will be returned as-is.
            - 'one-to-one': One key node can be connected to one value node.
            - 'one-to-many': One key node can be connected to multiple value
              nodes.
            - 'many-to-one': Multiple key nodes can be connected to one value
              node.
            - 'many-to-many': No restrictions on the number of edges that a
              key/value node can have.
        key_node_idx (int, optional): The label index of the key node. It must
            be specified if ``link_type`` is not "none". Defaults to None.
        value_node_idx (int, optional): The index of the value node. It must be
            specified if ``link_type`` is not "none". Defaults to None.
    """

    def __init__(self,
                 link_type: str = 'none',
                 key_node_idx: Optional[int] = None,
                 value_node_idx: Optional[int] = None):
        assert link_type in [
            'one-to-one', 'one-to-many', 'many-to-one', 'many-to-many', 'none'
        ]
        self.link_type = link_type
        if link_type != 'none':
            assert key_node_idx is not None and value_node_idx is not None
        self.key_node_idx = key_node_idx
        self.value_node_idx = value_node_idx
        self.softmax = nn.Softmax(dim=-1)

    def __call__(self, preds: Tuple[Tensor, Tensor],
                 data_samples: List[KIEDataSample]) -> List[KIEDataSample]:
        """Postprocess raw outputs from SDMGR heads and pack the results into a
        list of KIEDataSample.

        Args:
            preds (tuple[Tensor]): A tuple of raw outputs from SDMGR heads.
            data_samples (list[KIEDataSample]): A list of N datasamples,
                containing meta information and gold annotations for each of
                the images.

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
        node_preds, edge_preds = preds
        all_node_scores = self.softmax(node_preds)
        all_edge_scores = self.softmax(edge_preds)
        chunk_size = [
            data_sample.gt_instances.bboxes.shape[0]
            for data_sample in data_samples
        ]
        node_scores, node_preds = torch.max(all_node_scores, dim=-1)
        edge_scores, edge_preds = torch.max(all_edge_scores, dim=-1)
        node_preds = node_preds.split(chunk_size, dim=0)
        node_scores = node_scores.split(chunk_size, dim=0)

        sq_chunks = [chunk**2 for chunk in chunk_size]
        edge_preds = list(edge_preds.split(sq_chunks, dim=0))
        edge_scores = list(edge_scores.split(sq_chunks, dim=0))
        for i, chunk in enumerate(chunk_size):
            edge_preds[i] = edge_preds[i].reshape((chunk, chunk))
            edge_scores[i] = edge_scores[i].reshape((chunk, chunk))

        for i in range(len(data_samples)):
            data_samples[i].pred_instances = InstanceData()
            data_samples[i].pred_instances.labels = node_preds[i].cpu()
            data_samples[i].pred_instances.scores = node_scores[i].cpu()
            if self.link_type != 'none':
                edge_scores[i], edge_preds[i] = self.decode_edges(
                    node_preds[i], edge_scores[i], edge_preds[i])
            data_samples[i].pred_instances.edge_labels = edge_preds[i].cpu()
            data_samples[i].pred_instances.edge_scores = edge_scores[i].cpu()

        return data_samples

    def decode_edges(self, node_labels: Tensor, edge_scores: Tensor,
                     edge_labels: Tensor) -> Tuple[Tensor, Tensor]:
        """Reconstruct the edges and update edge scores according to
        ``link_type``.

        Args:
            data_sample (KIEDataSample): A datasample containing prediction
                results.

        Returns:
            tuple(Tensor, Tensor):

            - edge_scores (Tensor): A float tensor of shape (N, N)
                indicating the confidence scores for edge predictions.
            - edge_labels (Tensor): An integer tensor of shape (N, N)
                indicating the connection between nodes. Options are 0, 1.
        """
        # Obtain the scores of the existence of edges.
        pos_edges_scores = edge_scores.clone()
        edge_labels_mask = edge_labels.bool()
        pos_edges_scores[
            ~edge_labels_mask] = 1 - pos_edges_scores[~edge_labels_mask]

        # Temporarily convert the directed graph to undirected by adding
        # reversed edges to every pair of nodes if they were already connected
        # by an directed edge before.
        edge_labels = torch.max(edge_labels, edge_labels.T)

        # Maximize edge scores
        edge_labels_mask = edge_labels.bool()
        edge_scores[~edge_labels_mask] = pos_edges_scores[~edge_labels_mask]
        new_edge_scores = torch.max(edge_scores, edge_scores.T)

        # Only reconstruct the edges from key nodes to value nodes.
        key_nodes_mask = node_labels == self.key_node_idx
        value_nodes_mask = node_labels == self.value_node_idx
        key2value_mask = key_nodes_mask[:, None] * value_nodes_mask[None, :]

        if self.link_type == 'many-to-many':
            new_edge_labels = (key2value_mask * edge_labels).int()
        else:
            new_edge_labels = torch.zeros_like(edge_labels).int()

            tmp_edge_scores = new_edge_scores.clone().cpu()
            tmp_edge_scores[~edge_labels_mask] = -1
            tmp_edge_scores[~key2value_mask] = -1
            # Greedily extract valid edges
            while (tmp_edge_scores > -1).any():
                i, j = np.unravel_index(
                    torch.argmax(tmp_edge_scores), tmp_edge_scores.shape)
                new_edge_labels[i, j] = 1
                if self.link_type == 'one-to-one':
                    tmp_edge_scores[i, :] = -1
                    tmp_edge_scores[:, j] = -1
                elif self.link_type == 'one-to-many':
                    tmp_edge_scores[:, j] = -1
                elif self.link_type == 'many-to-one':
                    tmp_edge_scores[i, :] = -1

        return new_edge_scores.cpu(), new_edge_labels.cpu()
