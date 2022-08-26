_base_ = [
    '../_base_/datasets/mjsynth.py',
    '../_base_/datasets/synthtext.py',
    '../_base_/datasets/synthtext_add.py',
    '../_base_/datasets/coco_text_v1.py',
    '../_base_/datasets/cute80.py',
    '../_base_/datasets/iiit5k.py',
    '../_base_/datasets/svt.py',
    '../_base_/datasets/svtp.py',
    '../_base_/datasets/icdar2011.py',
    '../_base_/datasets/icdar2013.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_step_5e.py',
    '_base_robustscanner_resnet31.py',
]

default_hooks = dict(logger=dict(type='LoggerHook', interval=100))

# dataset settings
train_list = [
    _base_.ic11_rec_train, _base_.ic13_rec_train, _base_.ic15_rec_train,
    _base_.cocov1_rec_train, _base_.iiit5k_rec_train, _base_.mj_sub_rec_train,
    _base_.st_sub_rec_train, _base_.st_add_rec_train
]
test_list = [
    _base_.cute80_rec_test, _base_.iiit5k_rec_test, _base_.svt_rec_test,
    _base_.svtp_rec_test, _base_.ic13_rec_test, _base_.ic15_rec_test
]

train_list = [
    dict(
        type='RepeatDataset',
        dataset=dict(
            type='ConcatDataset',
            datasets=train_list[:5],
            pipeline=_base_.train_pipeline),
        times=20),
    dict(
        type='ConcatDataset',
        datasets=train_list[5:],
        pipeline=_base_.train_pipeline),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='ConcatDataset', datasets=train_list))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline,
        test_mode=True))
val_dataloader = test_dataloader

val_evaluator = dict(
    dataset_prefixes=['CUTE80', 'IIIT5K', 'SVT', 'SVTP', 'IC13', 'IC15'])
test_evaluator = val_evaluator
