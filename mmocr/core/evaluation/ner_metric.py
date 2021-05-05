from collections import Counter


def label2entity(gt_infos):
    """Get all entities from ground truth infos.
    Args:
        gt_infos (list[dict]): Groudtruth infomation contains text and label.
    Returns:
        gt_entities (list[list]): Original labeled entities in groundtruth.
                    [[category,start_position,end_position]]
    """
    gt_entities = []
    for gt_info in gt_infos:
        line_entities = []
        label = gt_info['label']
        for key, value in label.items():
            for _, places in value.items():
                for place in places:
                    line_entities.append([key, place[0], place[1]])
        gt_entities.append(line_entities)
    return gt_entities


def compute(origin, found, right):
    """Calculate recall, precision, f1.

    Args:
        origin: Original entities in groundtruth.
        found: Predicted entities from model.
        right: Predicted entities that can match to the original annotation.
    Returns:
        recall, precision, f1-score
    """
    recall = 0 if origin == 0 else (right / origin)
    precision = 0 if found == 0 else (right / found)
    f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (
        precision + recall)
    return recall, precision, f1


def pred_info(pred_entities, gt_entities):
    """Calculate precision, recall and F1-score for NER task.

    Args:
        pred_entities: The predicted entities from model.
        gt_entities: The entities of ground truth file.
    Returns:
        class_info (dict): precision,recall, f1-score in total and catogories.
    """
    origins = []
    founds = []
    rights = []
    for i, _ in enumerate(pred_entities):
        origins.extend(gt_entities[i])
        founds.extend(pred_entities[i])
        rights.extend([
            pre_entity for pre_entity in pred_entities[i]
            if pre_entity in gt_entities[i]
        ])

    class_info = {}
    origin_counter = Counter([x[0] for x in origins])
    found_counter = Counter([x[0] for x in founds])
    right_counter = Counter([x[0] for x in rights])
    for type_, count in origin_counter.items():
        origin = count
        found = found_counter.get(type_, 0)
        right = right_counter.get(type_, 0)
        recall, precision, f1 = compute(origin, found, right)
        class_info[type_] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4)
        }
    origin = len(origins)
    found = len(founds)
    right = len(rights)
    recall, precision, f1 = compute(origin, found, right)

    for key in sorted(class_info.keys()):
        print('******* %s results ********' % key)
        info = '-'.join([
            f' {key}: {value:.4f} ' for key, value in class_info[key].items()
        ])
        print(info)

    print({'precision': precision, 'recall': recall, 'f1': f1})
    return class_info


def eval_ner(results, gt_infos):
    """Evaluate for ner task.

    Args:
        results (list): Predict results of entities.
        gt_infos (list[dict]): Groudtruth infomation contains text and label .
    Returns:
        class_info (dict): precision,recall, f1-score of total
                            and each catogory.
    """
    assert len(results) == len(gt_infos)
    gt_entities = label2entity(gt_infos)
    pred_entities = []
    for i, gt_info in enumerate(gt_infos):
        text = gt_info['text']
        line_entities = []
        for result in results[i]:
            if result[2] < len(text) and result[1] < len(text):
                line_entities.append(result)
        pred_entities.append(line_entities)
    print('lens: {} vs {}'.format(len(pred_entities), len(gt_entities)))
    assert len(pred_entities) == len(gt_entities)
    class_info = pred_info(pred_entities, gt_entities)
    return class_info
