_base_ = [
    '../../_base_/schedules/schedule_sgd_600e.py',
    '../../_base_/runtime_10e.py',
    '../../_base_/det_models/psenet_r50_fpnf.py',
    '../../_base_/det_datasets/icdar2017.py',
    '../../_base_/det_pipelines/psenet_pipeline.py'
]

model = {{_base_.model_quad}}

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline_icdar2015 = {{_base_.test_pipeline_icdar2015}}

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_icdar2015),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_icdar2015))

evaluation = dict(interval=10, metric='hmean-iou')
