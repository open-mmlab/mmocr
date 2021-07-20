_base_ = [
    '../../_base_/schedules/schedule_1200e.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    type='DRRG',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='caffe'),
    neck=dict(
        type='FPN_UNet', in_channels=[256, 512, 1024, 2048], out_channels=32),
    bbox_head=dict(
        type='DRRGHead',
        in_channels=32,
        text_region_thr=0.3,
        center_region_thr=0.4,
        link_thr=0.80,
        loss=dict(type='DRRGLoss')))
train_cfg = None
test_cfg = None

dataset_type = 'IcdarDataset'
data_root = 'data/ctw1500/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ColorJitter', brightness=32.0 / 255, saturation=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomScaling', size=800, scale=(0.75, 2.5)),
    dict(
        type='RandomCropFlip', crop_ratio=0.5, iter_num=1, min_area_ratio=0.2),
    dict(
        type='RandomCropPolyInstances',
        instance_key='gt_masks',
        crop_ratio=0.8,
        min_side_ratio=0.3),
    dict(
        type='RandomRotatePolyInstances',
        rotate_ratio=0.5,
        max_angle=60,
        pad_with_fixed_color=False),
    dict(type='SquareResizePad', target_size=800, pad_ratio=0.6),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='DRRGTargets'),
    dict(type='Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=[
            'gt_text_mask', 'gt_center_region_mask', 'gt_mask',
            'gt_top_height_map', 'gt_bot_height_map', 'gt_sin_map',
            'gt_cos_map', 'gt_comp_attribs'
        ],
        visualize=dict(flag=False, boundary_key='gt_text_mask')),
    dict(
        type='Collect',
        keys=[
            'img', 'gt_text_mask', 'gt_center_region_mask', 'gt_mask',
            'gt_top_height_map', 'gt_bot_height_map', 'gt_sin_map',
            'gt_cos_map', 'gt_comp_attribs'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 640),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(1024, 640), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=f'{data_root}/instances_training.json',
        img_prefix=f'{data_root}/imgs',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=f'{data_root}/instances_test.json',
        img_prefix=f'{data_root}/imgs',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=f'{data_root}/instances_test.json',
        img_prefix=f'{data_root}/imgs',
        pipeline=test_pipeline))

evaluation = dict(interval=20, metric='hmean-iou')
