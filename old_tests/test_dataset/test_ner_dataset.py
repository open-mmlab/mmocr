# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import tempfile

import torch

from mmocr.datasets.ner_dataset import NerDataset
from mmocr.models.ner.convertors.ner_convertor import NerConvertor
from mmocr.utils import list_to_file


def _create_dummy_ann_file(ann_file):
    data = {
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

    list_to_file(ann_file, [json.dumps(data, ensure_ascii=False)])


def _create_dummy_vocab_file(vocab_file):
    for char in list(map(chr, range(ord('a'), ord('z') + 1))):
        list_to_file(vocab_file, [json.dumps(char + '\n', ensure_ascii=False)])


def _create_dummy_loader():
    loader = dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(type='LineJsonParser', keys=['text', 'label']))
    return loader


def test_ner_dataset():
    # test initialization
    loader = _create_dummy_loader()
    categories = [
        'address', 'book', 'company', 'game', 'government', 'movie', 'name',
        'organization', 'position', 'scene'
    ]

    # create dummy data
    tmp_dir = tempfile.TemporaryDirectory()
    ann_file = osp.join(tmp_dir.name, 'fake_data.txt')
    vocab_file = osp.join(tmp_dir.name, 'fake_vocab.txt')
    _create_dummy_ann_file(ann_file)
    _create_dummy_vocab_file(vocab_file)

    max_len = 128
    ner_convertor = dict(
        type='NerConvertor',
        annotation_type='bio',
        vocab_file=vocab_file,
        categories=categories,
        max_len=max_len)

    test_pipeline = [
        dict(
            type='NerTransform',
            label_convertor=ner_convertor,
            max_len=max_len),
        dict(type='ToTensorNER')
    ]
    dataset = NerDataset(ann_file, loader, pipeline=test_pipeline)

    # test pre_pipeline
    img_info = dataset.data_infos[0]
    results = dict(img_info=img_info)
    dataset.pre_pipeline(results)

    # test prepare_train_img
    dataset.prepare_train_img(0)

    # test evaluation
    result = [[['address', 15, 16], ['name', 0, 2]]]

    dataset.evaluate(result)

    # test pred convert2entity function
    pred = [
        21, 7, 17, 17, 21, 21, 21, 21, 21, 21, 13, 21, 21, 21, 21, 21, 1, 11,
        21, 21, 7, 17, 17, 21, 21, 21, 21, 21, 21, 13, 21, 21, 21, 21, 21, 1,
        11, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
        21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
        21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
        21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
        21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
        21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 1, 21, 21, 21, 21, 21,
        21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 1, 21, 21, 21, 21,
        21, 21
    ]
    preds = [pred[:128]]
    mask = [0] * 128
    for i in range(10):
        mask[i] = 1
    assert len(preds[0]) == len(mask)
    masks = torch.tensor([mask])
    convertor = NerConvertor(
        annotation_type='bio',
        vocab_file=vocab_file,
        categories=categories,
        max_len=128)
    all_entities = convertor.convert_pred2entities(preds=preds, masks=masks)
    assert len(all_entities[0][0]) == 3

    tmp_dir.cleanup()
