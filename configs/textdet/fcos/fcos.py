_base_ = [
    # '../../_base_/schedules/schedule_sgd_1200e.py',
    '../../_base_/runtime_10e.py',
    '../../_base_/det_datasets/icdar2015.py',
]
num_classes = 1
strides = [8, 16, 32, 64, 128]
bbox_coder = dict(type='DistancePointBBoxCoder')
with_bezier = False
norm_on_bbox = True
use_sigmoid_cls = True
model = dict(
    type='FCOS',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=strides,
        norm_on_bbox=norm_on_bbox,
        use_sigmoid_cls=use_sigmoid_cls,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        conv_bias=True,
        use_scale=False,
        with_bezier=with_bezier,
        init_cfg=dict(
            type='Normal',
            layer='Conv2d',
            std=0.01,
            override=dict(
                type='Normal',
                name='conv_cls',
                std=0.01,
                bias=--4.59511985013459),  # -log((1-p)/p) where p=0.01
        )),
    postprocessor=dict(
        type='ABCNetTextDetProcessor',
        strides=strides,
        bbox_coder=bbox_coder,
    ),
    loss=dict(
        type='FCOSLoss',
        num_classes=num_classes,
        strides=strides,
        center_sampling=True,
        center_sample_radius=1.5,
        bbox_coder=bbox_coder,
        with_bezier=with_bezier,
        norm_on_bbox=norm_on_bbox,
        use_sigmoid_cls=use_sigmoid_cls,
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=use_sigmoid_cls,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        rescale=True,
        property=['polygon', 'bboxes', 'bezier'],
        filter_and_location=True,
        reconstruct=True,
        extra_property=None,
        rescale_extra_property=False,
        nms_pre=1000,
        score_thr=0.3,
        strides=(8, 16, 32, 64, 128)))
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        # with_mask=True,
        # poly2mask=True,
        with_extra_fields=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'img',
            'gt_bboxes',
            'gt_labels',
            # 'gt_bezier_pts',
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                keys=[
                    'img',
                    'gt_bboxes',
                    'gt_labels',
                    # 'gt_bezier_pts',
                ]),
        ])
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
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
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
# runner = dict(type='EpochBasedRunner', max_epochs=12)
total_epochs = 12
