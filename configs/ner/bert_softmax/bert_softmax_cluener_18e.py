_base_ = [
    '../../_base_/schedules/schedule_adadelta_18e.py',
    '../../_base_/default_runtime.py'
]

categories = [
    'address', 'book', 'company', 'game', 'government', 'movie', 'name',
    'organization', 'position', 'scene'
]

test_ann_file = 'data/cluener_public/dev.json'
train_ann_file = 'data/cluener_public/train.json'
vocab_file = 'data/cluener_public/vocab.txt'
max_len = 128
loader = dict(
    type='HardDiskLoader',
    repeat=1,
    parser=dict(type='LineJsonParser', keys=['text', 'label']))

ner_convertor = dict(
    type='NerConvertor',
    annotation_type='bio',
    vocab_file=vocab_file,
    categories=categories,
    max_len=max_len)

test_pipeline = [
    dict(type='NerTransform', label_convertor=ner_convertor, max_len=max_len),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'texts', 'img', 'labels', 'input_ids', 'attention_mask',
            'token_type_ids'
        ])
]

train_pipeline = [
    dict(type='NerTransform', label_convertor=ner_convertor, max_len=max_len),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'texts', 'img', 'labels', 'input_ids', 'attention_mask',
            'token_type_ids'
        ]),
]
dataset_type = 'NerDataset'
img_prefix = ''

train = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=train_ann_file,
    loader=loader,
    pipeline=train_pipeline,
    test_mode=False)

test = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=test_ann_file,
    loader=loader,
    pipeline=test_pipeline,
    test_mode=True)
data = dict(
    samples_per_gpu=24,
    workers_per_gpu=2,
    train=dict(type='ConcatDataset', datasets=[train]),
    val=dict(type='ConcatDataset', datasets=[test]),
    test=dict(type='ConcatDataset', datasets=[test]))

evaluation = dict(interval=1, metric='acc')

model = dict(
    type='NerClassifier',
    encoder=dict(
        type='BertEncoder',
        num_hidden_layers=12,
        initializer_range=0.02,
        vocab_size=21128,
        hidden_size=768,
        max_position_embeddings=128,
        type_vocab_size=2,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.1,
        output_attentions=False,
        output_hidden_states=False,
        num_attention_heads=12,
        attention_probs_dropout_prob=0.1,
        intermediate_size=3072,
        hidden_act='gelu_new'),
    decoder=dict(type='FCDecoder', hidden_dropout_prob=0.1, hidden_size=768),
    loss=dict(type='NerLoss', loss_type='ce'),
    label_convertor=ner_convertor)

test_cfg = None
