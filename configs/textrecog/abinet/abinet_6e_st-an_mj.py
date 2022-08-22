_base_ = [
    '../../_base_/recog_datasets/mjsynth.py',
    '../../_base_/recog_datasets/synthtext.py',
    '../../_base_/recog_datasets/cute80.py',
    '../../_base_/recog_datasets/iiit5k.py',
    '../../_base_/recog_datasets/svt.py',
    '../../_base_/recog_datasets/svtp.py',
    '../../_base_/recog_datasets/icdar2013.py',
    '../../_base_/recog_datasets/icdar2015.py',
    '../../_base_/textrec_default_runtime.py',
    '../../_base_/schedules/schedule_adam_step_20e.py',
    '_base_abinet.py',
]

# dataset settings
train_list = [_base_.mj_rec_train, _base_.st_an_rec_train]
test_list = [
    _base_.cute80_rec_test, _base_.iiit5k_rec_test, _base_.svt_rec_test,
    _base_.svtp_rec_test, _base_.ic13_rec_test, _base_.ic15_rec_test
]

train_dataset = dict(
    type='ConcatDataset', datasets=train_list, pipeline=_base_.train_pipeline)
test_dataset = dict(
    type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline)

train_dataloader = dict(
    batch_size=192 * 4,
    num_workers=32,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

val_dataloader = test_dataloader

val_evaluator = dict(
    dataset_prefixes=['CUTE80', 'IIIT5K', 'SVT', 'SVTP', 'IC13', 'IC15'])
test_evaluator = val_evaluator
