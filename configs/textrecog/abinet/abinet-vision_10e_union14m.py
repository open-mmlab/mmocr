_base_ = [
    '../_base_/datasets/union14m_train.py',
    '../_base_/datasets/union14m_benchmark.py',
    '../_base_/datasets/cute80.py',
    '../_base_/datasets/iiit5k.py',
    '../_base_/datasets/svt.py',
    '../_base_/datasets/svtp.py',
    '../_base_/datasets/icdar2013.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adamw_cos_10e.py',
    '_base_abinet.py',
]

load_from = 'https://download.openmmlab.com/mmocr/textrecog/abinet/abinet_pretrain-45deac15.pth'  # noqa

_base_.pop('model')
dictionary = dict(
    type='Dictionary',
    dict_file=  # noqa
    '{{ fileDirname }}/../../../dicts/english_digits_symbols_space.txt',
    with_padding=True,
    with_unknown=True,
    same_start_end=True,
    with_start=True,
    with_end=True)

model = dict(
    type='ABINet',
    backbone=dict(type='ResNetABI'),
    encoder=dict(
        type='ABIEncoder',
        n_layers=3,
        n_head=8,
        d_model=512,
        d_inner=2048,
        dropout=0.1,
        max_len=8 * 32,
    ),
    decoder=dict(
        type='ABIFuser',
        vision_decoder=dict(
            type='ABIVisionDecoder',
            in_channels=512,
            num_channels=64,
            attn_height=8,
            attn_width=32,
            attn_mode='nearest',
            init_cfg=dict(type='Xavier', layer='Conv2d')),
        module_loss=dict(type='ABIModuleLoss'),
        postprocessor=dict(type='AttentionPostprocessor'),
        dictionary=dictionary,
        max_seq_len=26,
    ),
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]))

# dataset settings
train_list = [
    _base_.union14m_challenging, _base_.union14m_hard, _base_.union14m_medium,
    _base_.union14m_normal, _base_.union14m_easy
]
val_list = [
    _base_.cute80_textrecog_test, _base_.iiit5k_textrecog_test,
    _base_.svt_textrecog_test, _base_.svtp_textrecog_test,
    _base_.icdar2013_textrecog_test, _base_.icdar2015_textrecog_test
]
test_list = [
    _base_.union14m_benchmark_artistic,
    _base_.union14m_benchmark_multi_oriented,
    _base_.union14m_benchmark_contextless,
    _base_.union14m_benchmark_curve,
    _base_.union14m_benchmark_incomplete,
    _base_.union14m_benchmark_incomplete_ori,
    _base_.union14m_benchmark_multi_words,
    _base_.union14m_benchmark_salient,
    _base_.union14m_benchmark_general,
]

train_dataset = dict(
    type='ConcatDataset', datasets=train_list, pipeline=_base_.train_pipeline)
test_dataset = dict(
    type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline)
val_dataset = dict(
    type='ConcatDataset', datasets=val_list, pipeline=_base_.test_pipeline)

train_dataloader = dict(
    batch_size=128,
    num_workers=24,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

test_dataloader = dict(
    batch_size=128,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

val_dataloader = dict(
    batch_size=128,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset)

val_evaluator = dict(
    dataset_prefixes=['CUTE80', 'IIIT5K', 'SVT', 'SVTP', 'IC13', 'IC15'])
test_evaluator = dict(dataset_prefixes=[
    'artistic', 'multi-oriented', 'contextless', 'curve', 'incomplete',
    'incomplete-ori', 'multi-words', 'salient', 'general'
])
