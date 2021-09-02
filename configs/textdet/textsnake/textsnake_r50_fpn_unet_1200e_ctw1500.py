_base_ = [
    '../../_base_/schedules/schedule_1200e.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    type='TextSnake',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        norm_eval=True,
        style='caffe'),
    neck=dict(
        type='FPN_UNet', in_channels=[256, 512, 1024, 2048], out_channels=32),
    bbox_head=dict(
        type='TextSnakeHead',
        in_channels=32,
        text_repr_type='poly',
        loss=dict(type='TextSnakeLoss')),
    train_cfg=None,
    test_cfg=None)

dataset_type = 'IcdarDataset'
data_root = 'data/ctw1500/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(
        type='mmdet.LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ColorJitter', brightness=32.0 / 255, saturation=0.5),
    dict(type='mmdet.Normalize', **img_norm_cfg),
    dict(
        type='RandomCropPolyInstances',
        instance_key='gt_masks',
        crop_ratio=0.65,
        min_side_ratio=0.3),
    dict(
        type='RandomRotatePolyInstances',
        rotate_ratio=0.5,
        max_angle=20,
        pad_with_fixed_color=False),
    dict(
        type='ScaleAspectJitter',
        img_scale=[(3000, 736)],  # unused
        ratio_range=(0.7, 1.3),
        aspect_ratio_range=(0.9, 1.1),
        multiscale_mode='value',
        long_size_bound=800,
        short_size_bound=480,
        resize_type='long_short_bound',
        keep_ratio=False),
    dict(type='SquareResizePad', target_size=800, pad_ratio=0.6),
    dict(type='mmdet.RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='TextSnakeTargets'),
    dict(type='mmdet.Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=[
            'gt_text_mask', 'gt_center_region_mask', 'gt_mask',
            'gt_radius_map', 'gt_sin_map', 'gt_cos_map'
        ],
        visualize=dict(flag=False, boundary_key='gt_text_mask')),
    dict(
        type='mmdet.Collect',
        keys=[
            'img', 'gt_text_mask', 'gt_center_region_mask', 'gt_mask',
            'gt_radius_map', 'gt_sin_map', 'gt_cos_map'
        ])
]
test_pipeline = [
    dict(
        type='mmdet.LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='mmdet.MultiScaleFlipAug',
        img_scale=(1333, 736),
        flip=False,
        transforms=[
            dict(type='mmdet.Resize', img_scale=(1333, 736), keep_ratio=True),
            dict(type='mmdet.Normalize', **img_norm_cfg),
            dict(type='mmdet.Pad', size_divisor=32),
            dict(type='mmdet.ImageToTensor', keys=['img']),
            dict(type='mmdet.Collect', keys=['img']),
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

evaluation = dict(interval=10, metric='hmean-iou')
