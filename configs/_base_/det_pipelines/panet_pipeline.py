# avoid duplicate keys in _base_
from copy import deepcopy as pipeline_copy

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline_ctw1500 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ColorJitter', brightness=32.0 / 255, saturation=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(
        type='ScaleAspectJitter',
        img_scale=[(3000, 640)],
        ratio_range=(0.7, 1.3),
        aspect_ratio_range=(0.9, 1.1),
        multiscale_mode='value',
        keep_ratio=False),
    # shrink_ratio is from big to small. The 1st must be 1.0
    dict(type='PANetTargets', shrink_ratio=(1.0, 0.7)),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomRotateTextDet'),
    dict(
        type='RandomCropInstances',
        target_size=(640, 640),
        instance_key='gt_kernels'),
    dict(type='Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=['gt_kernels', 'gt_mask'],
        visualize=dict(flag=False, boundary_key='gt_kernels')),
    dict(type='Collect', keys=['img', 'gt_kernels', 'gt_mask'])
]

test_pipeline_ctw1500 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(3000, 640),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(3000, 640), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# for icdar2015
train_pipeline_icdar2015 = pipeline_copy(train_pipeline_ctw1500)
for pipeline in train_pipeline_icdar2015:
    if pipeline['type'] == 'ScaleAspectJitter':
        pipeline['img_scale'] = [(3000, 736)]
    if pipeline['type'] == 'PANetTargets':
        pipeline['shrink_ratio'] = (1.0, 0.5)
        pipeline['max_shrink'] = 20
    if pipeline['type'] == 'RandomCropInstances':
        pipeline['target_size'] = (736, 736)

test_pipeline_icdar2015 = pipeline_copy(test_pipeline_ctw1500)
for pipeline in test_pipeline_icdar2015:
    if pipeline['type'] == 'MultiScaleFlipAug':
        pipeline['img_scale'] = (1333, 736)

# for icdar2017
train_pipeline_icdar2017 = pipeline_copy(train_pipeline_ctw1500)
for pipeline in train_pipeline_icdar2017:
    if pipeline['type'] == 'ScaleAspectJitter':
        pipeline['img_scale'] = [(3000, 800)]
    if pipeline['type'] == 'PANetTargets':
        pipeline['shrink_ratio'] = (1.0, 0.5)
    if pipeline['type'] == 'RandomCropInstances':
        pipeline['target_size'] = (800, 800)

test_pipeline_icdar2017 = pipeline_copy(test_pipeline_ctw1500)
for pipeline in test_pipeline_icdar2017:
    if pipeline['type'] == 'MultiScaleFlipAug':
        pipeline['img_scale'] = (1333, 800)
