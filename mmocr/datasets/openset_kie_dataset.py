import copy

import numpy as np
import torch
from mmdet.datasets.builder import DATASETS

from mmocr.datasets import KIEDataset


@DATASETS.register_module()
class OpensetKIEDataset(KIEDataset):
    """Openset KIE classifies the nodes (i.e. text boxes) into bg/key/value
    categories, and additionally learns key-value relationship among nodes.

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        dict_file (str): Character dict file path.
        loader (dict): Dictionary to construct loader
            to load annotation infos.
        img_prefix (str, optional): Image prefix to generate full
            image path.
        test_mode (bool, optional): If True, try...except will
            be turned off in __getitem__.
        norm (float): Norm to map value from one range to another.
        link_type (str): ``one-to-one`` | ``one-to-many`` |
            ``many-to-one`` | ``many-to-many``. For ``many-to-many``,
            one key box can have many values and vice versa.
        edge_thr (float): Score threshold for a valid edge.
        test_mode (bool, optional): If True, try...except will
            be turned off in __getitem__.
        key_node_idx (int): Index of key in node classes.
        value_node_idx (int): Index of value in node classes.
        node_classes (int): Number of node classes.
    """

    def __init__(self,
                 ann_file,
                 loader,
                 dict_file,
                 img_prefix='',
                 pipeline=None,
                 norm=10.,
                 link_type='one-to-one',
                 edge_thr=0.5,
                 test_mode=True,
                 key_node_idx=1,
                 value_node_idx=2,
                 node_classes=4):
        super().__init__(ann_file, loader, dict_file, img_prefix, pipeline,
                         norm, False, test_mode)
        assert link_type in [
            'one-to-one', 'one-to-many', 'many-to-one', 'many-to-many', 'none'
        ]
        self.link_type = link_type
        self.data_dict = {x['file_name']: x for x in self.data_infos}
        self.edge_thr = edge_thr
        self.key_node_idx = key_node_idx
        self.value_node_idx = value_node_idx
        self.node_classes = node_classes

    def pre_pipeline(self, results):
        super().pre_pipeline(results)
        results['ori_texts'] = results['ann_info']['ori_texts']
        results['ori_boxes'] = results['ann_info']['ori_boxes']

    def list_to_numpy(self, ann_infos):
        results = super().list_to_numpy(ann_infos)
        results.update(dict(ori_texts=ann_infos['texts']))
        results.update(dict(ori_boxes=ann_infos['boxes']))

        return results

    def evaluate(self,
                 results,
                 metric='openset_f1',
                 metric_options=None,
                 **kwargs):
        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['openset_f1']
        for m in metrics:
            if m not in allowed_metrics:
                raise KeyError(f'metric {m} is not supported')

        preds, gts = [], []
        for result in results:
            # data for preds
            pred = self.decode_pred(result)
            preds.append(pred)
            # data for gts
            gt = self.decode_gt(pred['filename'])
            gts.append(gt)

        return self.compute_openset_f1(preds, gts)

    def _decode_pairs_gt(self, labels, edge_ids):
        """Find all pairs in gt.

        The first index in the pair (n1, n2) is key.
        """
        gt_pairs = []
        for i, label in enumerate(labels):
            if label == self.key_node_idx:
                for j, edge_id in enumerate(edge_ids):
                    if edge_id == edge_ids[i] and labels[
                            j] == self.value_node_idx:
                        gt_pairs.append((i, j))

        return gt_pairs

    @staticmethod
    def _decode_pairs_pred(nodes,
                           labels,
                           edges,
                           edge_thr=0.5,
                           link_type='one-to-one'):
        """Find all pairs in prediction.

        The first index in the pair (n1, n2) is more likely to be a key
        according to prediction in nodes.
        """
        edges = torch.max(edges, edges.T)
        if link_type in ['none', 'many-to-many']:
            pair_inds = (edges > edge_thr).nonzero(as_tuple=True)
            pred_pairs = [(n1.item(),
                           n2.item()) if nodes[n1, 1] > nodes[n1, 2] else
                          (n2.item(), n1.item()) for n1, n2 in zip(*pair_inds)
                          if n1 < n2]
            pred_pairs = [(i, j) for i, j in pred_pairs
                          if labels[i] == 1 and labels[j] == 2]
        else:
            links = edges.clone()
            links[links <= edge_thr] = -1
            links[labels != 1, :] = -1
            links[:, labels != 2] = -1

            pred_pairs = []
            while (links > -1).any():
                i, j = np.unravel_index(torch.argmax(links), links.shape)
                pred_pairs.append((i, j))
                if link_type == 'one-to-one':
                    links[i, :] = -1
                    links[:, j] = -1
                elif link_type == 'one-to-many':
                    links[:, j] = -1
                elif link_type == 'many-to-one':
                    links[i, :] = -1
                else:
                    raise ValueError(f'not supported link type {link_type}')

        pairs_conf = [edges[i, j].item() for i, j in pred_pairs]
        return pred_pairs, pairs_conf

    def decode_pred(self, result):
        """Decode prediction.

        Assemble boxes and predicted labels into bboxes, and convert edges into
        matrix.
        """
        filename = result['img_metas'][0]['ori_filename']
        nodes = result['nodes'].cpu()
        labels_conf, labels = torch.max(nodes, dim=-1)
        num_nodes = nodes.size(0)
        edges = result['edges'][:, -1].view(num_nodes, num_nodes).cpu()
        annos = self.data_dict[filename]['annotations']
        boxes = [x['box'] for x in annos]
        texts = [x['text'] for x in annos]
        bboxes = torch.Tensor(boxes)[:, [0, 1, 4, 5]]
        bboxes = torch.cat([bboxes, labels[:, None].float()], -1)
        pairs, pairs_conf = self._decode_pairs_pred(nodes, labels, edges,
                                                    self.edge_thr,
                                                    self.link_type)
        pred = {
            'filename': filename,
            'boxes': boxes,
            'bboxes': bboxes.tolist(),
            'labels': labels.tolist(),
            'labels_conf': labels_conf.tolist(),
            'texts': texts,
            'pairs': pairs,
            'pairs_conf': pairs_conf
        }
        return pred

    def decode_gt(self, filename):
        """Decode ground truth.

        Assemble boxes and labels into bboxes.
        """
        annos = self.data_dict[filename]['annotations']
        labels = torch.Tensor([x['label'] for x in annos])
        texts = [x['text'] for x in annos]
        edge_ids = [x['edge'] for x in annos]
        boxes = [x['box'] for x in annos]
        bboxes = torch.Tensor(boxes)[:, [0, 1, 4, 5]]
        bboxes = torch.cat([bboxes, labels[:, None].float()], -1)
        pairs = self._decode_pairs_gt(labels, edge_ids)
        gt = {
            'filename': filename,
            'boxes': boxes,
            'bboxes': bboxes.tolist(),
            'labels': labels.tolist(),
            'labels_conf': [1. for _ in labels],
            'texts': texts,
            'pairs': pairs,
            'pairs_conf': [1. for _ in pairs]
        }
        return gt

    def compute_openset_f1(self, preds, gts):
        """Compute openset macro-f1 and micro-f1 score.

        Args:
            preds: (list[dict]): List of prediction results, including
                keys: ``filename``, ``pairs``, etc.
            gts: (list[dict]): List of ground-truth infos, including
                keys: ``filename``, ``pairs``, etc.

        Returns:
            dict: Evaluation result with keys: ``node_openset_micro_f1``, \
                ``node_openset_macro_f1``, ``edge_openset_f1``.
        """

        total_edge_hit_num, total_edge_gt_num, total_edge_pred_num = 0, 0, 0
        total_node_hit_num, total_node_gt_num, total_node_pred_num = {}, {}, {}
        node_inds = list(range(self.node_classes))
        for node_idx in node_inds:
            total_node_hit_num[node_idx] = 0
            total_node_gt_num[node_idx] = 0
            total_node_pred_num[node_idx] = 0

        img_level_res = {}
        for pred, gt in zip(preds, gts):
            filename = pred['filename']
            img_res = {}
            # edge metric related
            pairs_pred = pred['pairs']
            pairs_gt = gt['pairs']
            img_res['edge_hit_num'] = 0
            for pair in pairs_gt:
                if pair in pairs_pred:
                    img_res['edge_hit_num'] += 1
            img_res['edge_recall'] = 1.0 * img_res['edge_hit_num'] / max(
                1, len(pairs_gt))
            img_res['edge_precision'] = 1.0 * img_res['edge_hit_num'] / max(
                1, len(pairs_pred))
            img_res['f1'] = 2 * img_res['edge_recall'] * img_res[
                'edge_precision'] / max(
                    1, img_res['edge_recall'] + img_res['edge_precision'])
            total_edge_hit_num += img_res['edge_hit_num']
            total_edge_gt_num += len(pairs_gt)
            total_edge_pred_num += len(pairs_pred)

            # node metric related
            nodes_pred = pred['labels']
            nodes_gt = gt['labels']
            for i, node_gt in enumerate(nodes_gt):
                node_gt = int(node_gt)
                total_node_gt_num[node_gt] += 1
                if nodes_pred[i] == node_gt:
                    total_node_hit_num[node_gt] += 1
            for node_pred in nodes_pred:
                total_node_pred_num[node_pred] += 1

            img_level_res[filename] = img_res

        stats = {}
        # edge f1
        total_edge_recall = 1.0 * total_edge_hit_num / max(
            1, total_edge_gt_num)
        total_edge_precision = 1.0 * total_edge_hit_num / max(
            1, total_edge_pred_num)
        edge_f1 = 2 * total_edge_recall * total_edge_precision / max(
            1, total_edge_recall + total_edge_precision)
        stats = {'edge_openset_f1': edge_f1}

        # node f1
        cared_node_hit_num, cared_node_gt_num, cared_node_pred_num = 0, 0, 0
        node_macro_metric = {}
        for node_idx in node_inds:
            if node_idx < 1 or node_idx > 2:
                continue
            cared_node_hit_num += total_node_hit_num[node_idx]
            cared_node_gt_num += total_node_gt_num[node_idx]
            cared_node_pred_num += total_node_pred_num[node_idx]
            node_res = {}
            node_res['recall'] = 1.0 * total_node_hit_num[node_idx] / max(
                1, total_node_gt_num[node_idx])
            node_res['precision'] = 1.0 * total_node_hit_num[node_idx] / max(
                1, total_node_pred_num[node_idx])
            node_res[
                'f1'] = 2 * node_res['recall'] * node_res['precision'] / max(
                    1, node_res['recall'] + node_res['precision'])
            node_macro_metric[node_idx] = node_res

        node_micro_recall = 1.0 * cared_node_hit_num / max(
            1, cared_node_gt_num)
        node_micro_precision = 1.0 * cared_node_hit_num / max(
            1, cared_node_pred_num)
        node_micro_f1 = 2 * node_micro_recall * node_micro_precision / max(
            1, node_micro_recall + node_micro_precision)

        stats['node_openset_micro_f1'] = node_micro_f1
        stats['node_openset_macro_f1'] = np.mean(
            [v['f1'] for k, v in node_macro_metric.items()])

        return stats
