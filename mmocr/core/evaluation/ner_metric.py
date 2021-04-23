import json
from collections import Counter


def get_entity_bio(seq, id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith('B-'):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entities(seq, id2label, markup='bios'):
    """Get entities.

    Args:
        seq (list): Sequence of labels.
        id2label (dict): Dict for mapping ID to label.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    """
    assert markup in ['bio', 'bios']
    if markup == 'bio':
        return get_entity_bio(seq, id2label)
    else:
        return get_entity_bios(seq, id2label)


def get_entity_bios(seq, id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith('S-'):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith('B-'):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


class SeqEntityScore(object):
    """Get precision, recall and F1-score for NER task.

    The code is adapted from https://github.com/lonePatient/BERT-NER-Pytorch
    """

    def __init__(self, id2label, markup='bios'):
        self.id2label = id2label
        self.markup = markup
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (
            precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([x[0] for x in self.origins])
        found_counter = Counter([x[0] for x in self.founds])
        right_counter = Counter([x[0] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {
                'acc': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4)
            }
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, label_paths, pred_paths):
        """
        Args:
            label_paths: [[],[],[],....]
            pred_paths:[[],[],[],.....]

        """
        for label_path, pre_path in zip(label_paths, pred_paths):
            label_entities = get_entities(label_path, self.id2label,
                                          self.markup)
            pre_entities = get_entities(pre_path, self.id2label, self.markup)
            self.origins.extend(label_entities)
            self.founds.extend(pre_entities)
            self.rights.extend([
                pre_entity for pre_entity in pre_entities
                if pre_entity in label_entities
            ])


def label2id(label, text_len, label2id_dict, max_len):
    id = [0] * max_len
    for j in range(text_len):
        id[j] = 31
    # text_len
    categorys = label
    for key in categorys:
        for text in categorys[key]:
            for place in categorys[key][text]:

                id[place[0]] = label2id_dict[key][0]

                for i in range(place[0] + 1, place[1] + 1):
                    id[i] = label2id_dict[key][1]
    return id


def eval_ner(res, gt, max_len, id2label, label2id_dict):
    """Evaluate for ner task.

    Args:
        res (list): Predict results.
        gt (list(dict)): Groudtruth file.
    """
    results = []
    for result in res:
        results.append(result[1:])
    with open(gt) as f:
        lines = f.readlines()
        metric = SeqEntityScore(id2label, markup='bios')
        for i in range(min(len(results), len(lines))):
            line_dict = json.loads(lines[i])
            text = line_dict['text']
            label = line_dict['label']
            label_ids = label2id(label, len(text), label2id_dict, max_len)
            # break
            pred = results[i]
            temp_1 = []
            temp_2 = []

            for j in range(len(label_ids)):  # enumerate(label_ids):
                if j == 0:
                    continue
                elif j == len(text) - 1:
                    metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break
                else:
                    temp_1.append(id2label[label_ids[j]])
                    temp_2.append(pred[j])

    eval_info, entity_info = metric.result()

    results = {f'{key}': value for key, value in eval_info.items()}

    info = '-'.join(
        [f' {key}: {value:.4f} ' for key, value in results.items()])
    print(info)
    for key in sorted(entity_info.keys()):
        print('******* %s results ********' % key)
        info = '-'.join([
            f' {key}: {value:.4f} ' for key, value in entity_info[key].items()
        ])
        print(info)
    return eval_info
