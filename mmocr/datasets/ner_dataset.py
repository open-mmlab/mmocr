import json

from mmdet.datasets.builder import DATASETS
from mmocr.core.evaluation.ner_metric import eval_ner
from mmocr.datasets.base_dataset import BaseDataset


@DATASETS.register_module()
class NerDataset(BaseDataset):
    """
    Args:
        ann_file (txt): Annotation file path.
        loader (dict): Dictionary to construct loader
            to load annotation infos.
        pipeline (list[dict]): Processing pipeline.
        img_prefix (str, optional): This parameter is not used.
        test_mode (bool, optional): If True, try...except will
            be turned off in __getitem__.
        vocab_file (str): File to convert words to ids.
        map_file (str): File to get label2id_dict and word2ids_dict.
        max_len (int): The maximum reserved length of the input.
        unknown_id (int): All characters that do not appear in
            vocab_file are marked as unknown_id.
        start_id (int): An ID added before each input text.
        end_id (int): An ID added after each input text.
    """

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
        self.ignore_label = None
        with open(self.map_file, 'r') as f:
            map_dict = json.load(f)
            self.label2id_dict = map_dict['label2id_dict']
            id2label = map_dict['id2label']
            self.id2label = {}
            for key, value in id2label.items():
                self.id2label.update({int(key): value})
                if value == 'O':
                    self.ignore_label = int(key)
        assert self.ignore_label is not None
        lines = open(vocab_file, encoding='utf-8').readlines()
        for i in range(len(lines)):
            self.word2ids.update({lines[i].rstrip(): i})

    def _convert_text2id(self, text):
        """Convert characters to ids.

        If the input is uppercase,
            convert to lowercase first.
        Args:
            text (list[char]): Annotations of one paragraph.
        Returns:
            ids (list): Corresponding IDs after conversion.
        """
        ids = []
        for word in text.lower():
            if word in self.word2ids:
                ids.append(self.word2ids[word])
            else:
                ids.append(self.unknown_id)
        return ids

    def _conver_entity2label(self, label, text_len):
        """Convert labeled entities to ids.

        Args:
            label (dict): Labels of entities.
            text_len (int): The length of input text.
        Returns:
            labels (list): Label ids of an input text.
        """
        labels = [0] * self.max_len
        for j in range(text_len + 2):
            labels[j] = self.ignore_label
        categorys = label
        for key in categorys:
            for text in categorys[key]:
                for place in categorys[key][text]:
                    labels[place[0] + 1] = self.label2id_dict[key][0]

                    for i in range(place[0] + 1, place[1] + 1):
                        labels[i + 1] = self.label2id_dict[key][1]
        return labels

    def _parse_anno_info(self, ann):
        """Parse annotations of texts and labels for one text.
        Args:
            ann (dict): Annotations of texts and labels for one text
        Returns:
            ans: A dict containing the following keys:
                img, labels, texts, input_ids, attention_mask and token_type_ids.
        """

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
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            info (dict): A dict containing the following keys: 'acc', 'recall', 'f1'.
        """
        gt = self.ann_file
        info = eval_ner(results, gt, self.max_len, self.id2label,
                        self.label2id_dict, self.ignore_label)
        return info
