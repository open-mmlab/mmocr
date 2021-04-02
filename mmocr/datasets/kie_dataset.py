import copy
from os import path as osp

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmocr.core import compute_f1_score


@DATASETS.register_module()
class KIEDataset(CustomDataset):

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 img_prefix='',
                 ann_prefix='',
                 vocab_file=None,
                 class_file=None,
                 norm=10.,
                 thresholds=dict(edge=0.5),
                 directed=False,
                 **kwargs):
        self.ann_prefix = ann_prefix
        self.norm = norm
        self.thresholds = thresholds
        self.directed = directed

        if data_root is not None:
            if not osp.isabs(ann_file):
                self.ann_file = osp.join(data_root, ann_file)
            if not (ann_prefix is None or osp.isabs(ann_prefix)):
                self.ann_prefix = osp.join(data_root, ann_prefix)

        self.vocab = dict({'': 0})
        vocab_file = osp.join(data_root, vocab_file)
        if osp.exists(vocab_file):
            with open(vocab_file, 'r') as fid:
                for idx, char in enumerate(fid.readlines(), 1):
                    self.vocab[char.strip('\n')] = idx
        else:
            self.construct_dict(self.ann_file)
            with open(vocab_file, 'w') as fid:
                for key in self.vocab:
                    if key:
                        fid.write('{}\n'.format(key))

        super().__init__(
            ann_file,
            pipeline,
            data_root=data_root,
            img_prefix=img_prefix,
            **kwargs)

        self.idx_to_cls = dict()
        with open(osp.join(data_root, class_file), 'r') as fid:
            for line in fid.readlines():
                idx, cls = line.split()
                self.idx_to_cls[int(idx)] = cls

    @staticmethod
    def _split_edge(line):
        text = ','.join(line[8:-1])
        if ';' in text and text.split(';')[0].isdecimal():
            edge, text = text.split(';', 1)
            edge = int(edge)
        else:
            edge = 0
        return edge, text

    def construct_dict(self, ann_file):
        img_infos = mmcv.list_from_file(ann_file)
        for img_info in img_infos:
            _, annname = img_info.split()
            if self.ann_prefix:
                annname = osp.join(self.ann_prefix, annname)
            with open(annname, 'r') as fid:
                lines = fid.readlines()

            for line in lines:
                line = line.strip().split(',')
                _, text = self._split_edge(line)
                for c in text:
                    if c not in self.vocab:
                        self.vocab[c] = len(self.vocab)
        self.vocab = dict(
            {k: idx
             for idx, k in enumerate(sorted(self.vocab.keys()))})

    def convert_text(self, text):
        return [self.vocab[c] for c in text if c in self.vocab]

    def parse_lines(self, annname):
        boxes, edges, texts, chars, labels = [], [], [], [], []

        if self.ann_prefix:
            annname = osp.join(self.ann_prefix, annname)

        with open(annname, 'r') as fid:
            for line in fid.readlines():
                line = line.strip().split(',')
                boxes.append(list(map(int, line[:8])))
                edge, text = self._split_edge(line)
                chars.append(text)
                text = self.convert_text(text)
                texts.append(text)
                edges.append(edge)
                labels.append(int(line[-1]))
        return dict(
            boxes=boxes, edges=edges, texts=texts, chars=chars, labels=labels)

    def format_results(self, results):
        boxes = torch.Tensor(results['boxes'])[:, [0, 1, 4, 5]].cuda()

        if 'nodes' in results:
            nodes, edges = results['nodes'], results['edges']
            labels = nodes.argmax(-1)
            num_nodes = nodes.size(0)
            edges = edges[:, -1].view(num_nodes, num_nodes)
        else:
            labels = torch.Tensor(results['labels']).cuda()
            edges = torch.Tensor(results['edges']).cuda()
        boxes = torch.cat([boxes, labels[:, None].float()], -1)

        return {
            **{
                k: v
                for k, v in results.items() if k not in ['boxes', 'edges']
            }, 'boxes': boxes,
            'edges': edges,
            'points': results['boxes']
        }

    def plot(self, results):
        img_name = osp.join(self.img_prefix, results['filename'])
        img = plt.imread(img_name)
        plt.imshow(img)

        boxes, texts = results['points'], results['chars']
        num_nodes = len(boxes)
        if 'scores' in results:
            scores = results['scores']
        else:
            scores = np.ones(num_nodes)
        for box, text, score in zip(boxes, texts, scores):
            xs, ys = [], []
            for idx in range(0, 10, 2):
                xs.append(box[idx % 8])
                ys.append(box[(idx + 1) % 8])
            plt.plot(xs, ys, 'g')
            plt.annotate(
                '{}: {:.4f}'.format(text, score), (box[0], box[1]), color='g')

        if 'nodes' in results:
            nodes = results['nodes']
            inds = nodes.argmax(-1)
        else:
            nodes = np.ones((num_nodes, 3))
            inds = results['labels']
        for i in range(num_nodes):
            plt.annotate(
                '{}: {:.4f}'.format(
                    self.idx_to_cls(inds[i] - 1), nodes[i, inds[i]]),
                (boxes[i][6], boxes[i][7]),
                color='r' if inds[i] == 1 else 'b')
            edges = results['edges']
            if 'nodes' not in results:
                edges = (edges[:, None] == edges[None]).float()
            for j in range(i + 1, num_nodes):
                edge_score = max(edges[i][j], edges[j][i])
                if edge_score > self.thresholds['edge']:
                    x1 = sum(boxes[i][:3:2]) // 2
                    y1 = sum(boxes[i][3:6:2]) // 2
                    x2 = sum(boxes[j][:3:2]) // 2
                    y2 = sum(boxes[j][3:6:2]) // 2
                    plt.plot((x1, x2), (y1, y2), 'r')
                    plt.annotate(
                        '{:.4f}'.format(edge_score),
                        ((x1 + x2) // 2, (y1 + y2) // 2),
                        color='r')

    def compute_relation(self, boxes):
        x1s, y1s = boxes[:, 0:1], boxes[:, 1:2]
        x2s, y2s = boxes[:, 4:5], boxes[:, 5:6]
        ws, hs = x2s - x1s + 1, np.maximum(y2s - y1s + 1, 1)
        dxs = (x1s[:, 0][None] - x1s) / self.norm
        dys = (y1s[:, 0][None] - y1s) / self.norm
        xhhs, xwhs = hs[:, 0][None] / hs, ws[:, 0][None] / hs
        whs = ws / hs + np.zeros_like(xhhs)
        relations = np.stack([dxs, dys, whs, xhhs, xwhs], -1)
        bboxes = np.concatenate([x1s, y1s, x2s, y2s], -1).astype(np.float32)
        return relations, bboxes

    def ann_numpy(self, results):
        boxes, texts = results['boxes'], results['texts']
        boxes = np.array(boxes, np.int32)
        if boxes[0, 1] > boxes[0, -1]:
            boxes = boxes[:, [6, 7, 4, 5, 2, 3, 0, 1]]
        relations, bboxes = self.compute_relation(boxes)

        labels = results.get('labels', None)
        if labels is not None:
            labels = np.array(labels, np.int32)
            edges = results.get('edges', None)
            if edges is not None:
                labels = labels[:, None]
                edges = np.array(edges)
                edges = (edges[:, None] == edges[None, :]).astype(np.int32)
                if self.directed:
                    edges = (edges & labels == 1).astype(np.int32)
                np.fill_diagonal(edges, -1)
                labels = np.concatenate([labels, edges], -1)
        return dict(
            bboxes=bboxes,
            relations=relations,
            texts=self.pad_text(texts),
            labels=labels)

    def image_size(self, filename):
        img_path = osp.join(self.img_prefix, filename)
        img = Image.open(img_path)
        return img.size

    def load_annotations(self, ann_file):
        self.anns, data_infos = [], []

        self.gts = dict()
        img_infos = mmcv.list_from_file(ann_file)
        for img_info in img_infos:
            filename, annname = img_info.split()
            results = self.parse_lines(annname)
            width, height = self.image_size(filename)

            data_infos.append(
                dict(filename=filename, width=width, height=height))
            ann = self.ann_numpy(results)
            self.anns.append(ann)

        return data_infos

    def pad_text(self, texts):
        max_len = max([len(text) for text in texts])
        padded_texts = -np.ones((len(texts), max_len), np.int32)
        for idx, text in enumerate(texts):
            padded_texts[idx, :len(text)] = np.array(text)
        return padded_texts

    def get_ann_info(self, idx):
        return self.anns[idx]

    def prepare_test_img(self, idx):
        return self.prepare_train_img(idx)

    def evaluate(self,
                 results,
                 metric='macro_f1',
                 metric_options=dict(macro_f1=dict(ignores=[])),
                 **kwargs):
        # allow some kwargs to pass through
        assert set(kwargs).issubset(['logger'])

        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['macro_f1']
        for m in metrics:
            if m not in allowed_metrics:
                raise KeyError(f'metric {m} is not supported')

        return self.compute_macro_f1(results, **metric_options['macro_f1'])

    def compute_macro_f1(self, results, ignores=[]):
        node_preds = []
        for result in results:
            node_preds.append(result['nodes'])
        node_preds = torch.cat(node_preds)

        node_gts = [
            torch.from_numpy(ann['labels'][:, 0]).to(node_preds.device)
            for ann in self.anns
        ]
        node_gts = torch.cat(node_gts)

        node_f1s = compute_f1_score(node_preds, node_gts, ignores)

        return {
            'macro_f1': node_f1s.mean(),
        }
