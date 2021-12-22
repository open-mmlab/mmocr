# Copyright (c) OpenMMLab. All rights reserved.
_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/recog_pipelines/seg_pipeline.py',
    '../../_base_/recog_models/seg.py',
    '../../_base_/recog_datasets/ST_charbox_train.py',
    '../../_base_/recog_datasets/academic_test.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}

# optimizer
optimizer = dict(type='Adam', lr=1e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[3, 4])
total_epochs = 5

find_unused_parameters = True

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='acc')
