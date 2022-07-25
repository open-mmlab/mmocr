_base_ = [
    'panet_r50_fpem_ffm.py', '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_adam_600e.py',
    '../../_base_/det_datasets/icdar2017.py',
    '../../_base_/det_pipelines/panet_pipeline.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline_icdar2017 = {{_base_.train_pipeline_icdar2017}}
test_pipeline_icdar2017 = {{_base_.test_pipeline_icdar2017}}

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline_icdar2017),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_icdar2017),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_icdar2017))

evaluation = dict(interval=10, metric='hmean-iou')
