import json

from mmocr.datasets.ner_dataset import NerDataset
import pytest

def _create_dummy_gt_file(ann_file):
    dict_str = {
        'text': '彭小军认为，国内银行现在走的是台湾的发卡模式',
        'label': {
            'address': {
                '台湾': [[15, 16]]
            },
            'name': {
                '彭小军': [[0, 2]]
            }
        }
    }
    with open(ann_file, 'w') as fw:
        fw.write(json.dumps(dict_str, ensure_ascii=False) + '\n')
    return ann_file


def _create_dummy_loader():
    loader = dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(type='LineJsonParser', keys=['text', 'label']))
    return loader


def test_ner_dataset():
    # test initialization
    loader = _create_dummy_loader()

    ann_file = 'tests/data/ner_dataset/dev.json'
    vocab_file = 'tests/data/ner_dataset/vocab.txt'
    map_file = 'tests/data/ner_dataset/map_file.json'
    test_pipeline = [
        dict(
            type='Collect',
            keys=['img'],
            meta_keys=[
                'texts', 'img', 'labels', 'input_ids', 'attention_mask',
                'token_type_ids'
            ])
    ]
    dataset = NerDataset(
        ann_file,
        loader,
        pipeline=test_pipeline,
        vocab_file=vocab_file,
        map_file=map_file,
    )

    # test pre_pipeline
    img_info = dataset.data_infos[0]
    results = dict(img_info=img_info)
    dataset.pre_pipeline(results)
    #test _parse_anno_info
    ann={"text": "彭小军认为，国内银行现在走的是台湾的发卡模式",
     "label": {"address": {"台湾": [[15, 16]]}, "name": {"彭小军": [[0, 2]]}}}

    ans = dataset._parse_anno_info(ann)
    assert isinstance(ans, dict)
    assert "img" in ans
    #test prepare_train_img
    dataset.prepare_train_img(0)

    # test evaluation
    result = []
    result.append([
        31,7, 17, 17, 31, 31, 31, 31, 31, 31, 13, 31, 31, 31, 31, 31, 1, 11, 31,
        31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
        31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
        31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
        31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
        31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
        31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 1, 31, 31, 31, 31, 31, 31
    ])
    dataset.evaluate(result)
