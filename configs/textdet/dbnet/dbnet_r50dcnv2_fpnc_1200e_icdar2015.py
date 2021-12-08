_base_ = [
    '../../_base_/runtime_10e.py',
    '../../_base_/schedules/schedule_sgd_1200e.py',
    '../../_base_/det_models/dbnet_r50dcnv2_fpnc.py',
    '../../_base_/det_datasets/icdar2015.py',
    '../../_base_/det_pipelines/dbnet_pipeline.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline_r50dcnv2 = {{_base_.train_pipeline_r50dcnv2}}
test_pipeline_4068_1024 = {{_base_.test_pipeline_4068_1024}}

load_from = 'checkpoints/textdet/dbnet/res50dcnv2_synthtext.pth'

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline_r50dcnv2),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_4068_1024),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_4068_1024))

evaluation = dict(interval=100, metric='hmean-iou')
