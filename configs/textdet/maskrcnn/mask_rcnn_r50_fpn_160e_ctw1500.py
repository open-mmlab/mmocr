# Copyright (c) OpenMMLab. All rights reserved.
_base_ = [
    '../../_base_/runtime_10e.py',
    '../../_base_/det_models/ocr_mask_rcnn_r50_fpn_ohem_poly.py',
    '../../_base_/schedules/schedule_sgd_160e.py',
    '../../_base_/det_datasets/ctw1500.py',
    '../../_base_/det_pipelines/maskrcnn_pipeline.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline_ctw1500 = {{_base_.test_pipeline_ctw1500}}

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
        pipeline=test_pipeline_ctw1500),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_ctw1500))

evaluation = dict(interval=10, metric='hmean-iou')
