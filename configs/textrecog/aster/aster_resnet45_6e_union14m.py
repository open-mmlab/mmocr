# training schedule for 1x
_base_ = [
    '_base_aster.py',
    '../_base_/datasets/union14m_train.py',
    '../_base_/datasets/union14m_benchmark.py',
    '../_base_/datasets/cute80.py',
    '../_base_/datasets/iiit5k.py',
    '../_base_/datasets/svt.py',
    '../_base_/datasets/svtp.py',
    '../_base_/datasets/icdar2013.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adamw_cos_6e.py',
]

dictionary = dict(
    type='Dictionary',
    dict_file=  # noqa
    '{{ fileDirname }}/../../../dicts/english_digits_symbols_space.txt',
    with_padding=True,
    with_unknown=True,
    same_start_end=True,
    with_start=True,
    with_end=True)

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

default_hooks = dict(logger=dict(type='LoggerHook', interval=50))

auto_scale_lr = dict(base_batch_size=512)

train_dataset = dict(
    type='ConcatDataset', datasets=train_list, pipeline=_base_.train_pipeline)
test_dataset = dict(
    type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline)
val_dataset = dict(
    type='ConcatDataset', datasets=val_list, pipeline=_base_.test_pipeline)

train_dataloader = dict(
    batch_size=512,
    num_workers=12,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

test_dataloader = dict(
    batch_size=128,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
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
