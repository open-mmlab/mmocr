from mmocr.datasets.ner_dataset import NerDataset
from mmocr.models.ner.convertors.ner_convertor import NerConvertor


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
    ann_file = 'tests/data/ner_toy_dataset/train_sample.json'
    vocab_file = 'tests/data/ner_toy_dataset/vocab_sample.txt'
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
        dict(
            type='Collect',
            keys=['img'],
            meta_keys=[
                'texts', 'img', 'labels', 'input_ids', 'attention_mask',
                'token_type_ids'
            ])
    ]
    dataset = NerDataset(ann_file, loader, pipeline=test_pipeline)

    # test pre_pipeline
    img_info = dataset.data_infos[0]
    results = dict(img_info=img_info)
    dataset.pre_pipeline(results)
    # test _parse_anno_info
    ann = {
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

    # test prepare_train_img
    dataset.prepare_train_img(0)

    # test evaluation
    result = [[['address', 15, 16], ['name', 0, 2]]]

    dataset.evaluate(result)

    # test pred convert2entity function
    pred = [[
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
    ]]
    convertor = NerConvertor(
        annotation_type='bio',
        vocab_file=vocab_file,
        categories=categories,
        max_len=128)
    all_entities = convertor.convert_pred2entities(preds=pred)
    assert len(all_entities[0][0]) == 3
