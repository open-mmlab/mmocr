_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/recog_datasets/seg_toy_data.py',
    '../../_base_/recog_models/seg.py',
    '../../_base_/recog_pipelines/seg_pipeline.py',
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

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
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

find_unused_parameters = True
