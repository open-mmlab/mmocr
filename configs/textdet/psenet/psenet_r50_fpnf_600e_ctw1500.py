_base_ = ['../../_base_/default_runtime.py']

# optimizer
optimizer = dict(type='Adam', lr=1e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[200, 400])
total_epochs = 600

model = dict(
    type='PSENet',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        norm_eval=True,
        style='caffe'),
    neck=dict(
        type='FPNF',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        fusion_type='concat'),
    bbox_head=dict(
        type='PSEHead',
        text_repr_type='poly',
        in_channels=[256],
        out_channels=7,
        loss=dict(type='PSELoss')),
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
        type='ScaleAspectJitter',
        img_scale=[(3000, 736)],
        ratio_range=(0.5, 3),
        aspect_ratio_range=(1, 1),
        multiscale_mode='value',
        long_size_bound=1280,
        short_size_bound=640,
        resize_type='long_short_bound',
        keep_ratio=False),
    dict(type='PSENetTargets'),
    dict(type='mmdet.RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomRotateTextDet'),
    dict(
        type='RandomCropInstances',
        target_size=(640, 640),
        instance_key='gt_kernels'),
    dict(type='mmdet.Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=['gt_kernels', 'gt_mask'],
        visualize=dict(flag=False, boundary_key='gt_kernels')),
    dict(type='mmdet.Collect', keys=['img', 'gt_kernels', 'gt_mask'])
]
test_pipeline = [
    dict(
        type='mmdet.LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='mmdet.MultiScaleFlipAug',
        img_scale=(1280, 1280),
        flip=False,
        transforms=[
            dict(type='mmdet.Resize', img_scale=(1280, 1280), keep_ratio=True),
            dict(type='mmdet.Normalize', **img_norm_cfg),
            dict(type='mmdet.Pad', size_divisor=32),
            dict(type='mmdet.ImageToTensor', keys=['img']),
            dict(type='mmdet.Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/instances_training.json',
        img_prefix=data_root + '/imgs',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/instances_test.json',
        img_prefix=data_root + '/imgs',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/instances_test.json',
        img_prefix=data_root + '/imgs',
        pipeline=test_pipeline))

evaluation = dict(interval=10, metric='hmean-iou')
