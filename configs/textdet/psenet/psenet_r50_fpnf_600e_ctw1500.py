_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_adam_step_600e.py',
    '../../_base_/det_models/psenet_r50_fpnf.py',
    '../../_base_/det_datasets/ctw1500.py',
    '../../_base_/det_pipelines/psenet_pipeline.py'
]

model = {{_base_.model_poly}}

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline_ctw1500 = {{_base_.test_pipeline_ctw1500}}

data = dict(
    samples_per_gpu=2,
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
        pipeline=test_pipeline_ctw1500),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_ctw1500))

evaluation = dict(interval=10, metric='hmean-iou')
