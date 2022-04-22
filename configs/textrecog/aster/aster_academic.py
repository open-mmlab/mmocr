_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/recog_pipelines/aster_pipeline.py',
    '../../_base_/recog_datasets/aster_academic_train.py',
    '../../_base_/recog_datasets/academic_test.py',
    '../../_base_/schedules/schedule_adadelta_5e.py',
    '../../_base_/recog_models/asternet.py'
]


train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
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

evaluation = dict(interval=100, metric='acc')