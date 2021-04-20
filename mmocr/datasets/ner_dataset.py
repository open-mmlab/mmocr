import json

from mmdet.datasets.builder import DATASETS
from mmocr.core.evaluation.ner import eval_ner
from mmocr.datasets.base_dataset import BaseDataset


@DATASETS.register_module()
class NerDataset(BaseDataset):

    def __init__(self,
                 ann_file,
                 loader,
                 pipeline,
                 img_prefix='',
                 test_mode=False,
                 vocab_file=None,
                 map_file=None,
                 max_len=128,
                 unknown_id=100,
                 start_id=101,
                 end_id=102):
        super().__init__(
            ann_file, loader, pipeline, img_prefix='', test_mode=False)
        self.ann_file = ann_file
        self.word2ids = {}
        self.unknown_id = unknown_id
        self.start_id = start_id
        self.end_id = end_id
        self.max_len = max_len
        self.map_file = map_file
        with open(self.map_file) as f:
            map_dict = json.load(f)
            self.label2id_dict = map_dict['label2id_dict']
            id2label = map_dict['id2label']
            self.id2label = {}
            for key, value in id2label.items():
                self.id2label.update({int(key): value})
        lines = open(vocab_file, encoding='utf-8').readlines()
        for i in range(len(lines)):
            self.word2ids.update({lines[i].rstrip(): i})

    def _convert_text2id(self, text):
        ids = []
        for word in text.lower():
            if word in self.word2ids:
                ids.append(self.word2ids[word])
            else:
                ids.append(self.unknown_id)
        return ids

    def _conver_entity2label(self, label, text_len):
        labels = [0] * self.max_len
        for j in range(text_len + 2):
            labels[j] = 31
        categorys = label
        for key in categorys:
            for text in categorys[key]:
                for place in categorys[key][text]:
                    labels[place[0] + 1] = self.label2id_dict[key][0]

                    for i in range(place[0] + 1, place[1] + 1):
                        labels[i + 1] = self.label2id_dict[key][1]
        return labels

    def _parse_anno_info(self, ann):

        ids = self._convert_text2id(ann['text'])
        labels = self._conver_entity2label(ann['label'], len(ann['text']))
        texts = ann['text']

        valid_len = len(texts)
        use_len = len(labels)

        input_ids = [0] * use_len
        attention_mask = [0] * use_len
        token_type_ids = [0] * use_len

        input_ids[0] = self.start_id
        attention_mask[0] = 1
        for i in range(1, valid_len + 1):
            input_ids[i] = ids[i - 1]
            attention_mask[i] = 1
        input_ids[i + 1] = self.end_id
        attention_mask[i + 1] = 1
        ans = dict(
            img=ids,
            labels=labels,
            texts=texts,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)
        return ans

    def prepare_train_img(self, index):
        """Get training data and annotations after pipeline.

        Args:
            index (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        img_ann_info = self.data_infos[index]
        ann_info = self._parse_anno_info(img_ann_info)
        results = dict(ann_info)
        return self.pipeline(results)

    def evaluate(self, results, metric=None, logger=None, **kwargs):
        # import pdb
        # pdb.set_trace()
        gt = self.ann_file
        info = eval_ner(results, gt, self.max_len, self.id2label,
                        self.label2id_dict)
        return info
