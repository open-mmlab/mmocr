img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# for ctw1500
img_scale_train_ctw1500 = [(3000, 640)]
shrink_ratio_train_ctw1500 = (1.0, 0.7)
target_size_train_ctw1500 = (640, 640)
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
        img_scale=img_scale_train_ctw1500,
        ratio_range=(0.7, 1.3),
        aspect_ratio_range=(0.9, 1.1),
        multiscale_mode='value',
        keep_ratio=False),
    # shrink_ratio is from big to small. The 1st must be 1.0
    dict(type='PANetTargets', shrink_ratio=shrink_ratio_train_ctw1500),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomRotateTextDet'),
    dict(
        type='RandomCropInstances',
        target_size=target_size_train_ctw1500,
        instance_key='gt_kernels'),
    dict(type='Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=['gt_kernels', 'gt_mask'],
        visualize=dict(flag=False, boundary_key='gt_kernels')),
    dict(type='Collect', keys=['img', 'gt_kernels', 'gt_mask'])
]

img_scale_test_ctw1500 = (3000, 640)
test_pipeline_ctw1500 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale_test_ctw1500,  # used by Resize
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# for icdar2015
img_scale_train_icdar2015 = [(3000, 736)]
shrink_ratio_train_icdar2015 = (1.0, 0.5)
target_size_train_icdar2015 = (736, 736)
train_pipeline_icdar2015 = [
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
        img_scale=img_scale_train_icdar2015,
        ratio_range=(0.7, 1.3),
        aspect_ratio_range=(0.9, 1.1),
        multiscale_mode='value',
        keep_ratio=False),
    dict(type='PANetTargets', shrink_ratio=shrink_ratio_train_icdar2015),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomRotateTextDet'),
    dict(
        type='RandomCropInstances',
        target_size=target_size_train_icdar2015,
        instance_key='gt_kernels'),
    dict(type='Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=['gt_kernels', 'gt_mask'],
        visualize=dict(flag=False, boundary_key='gt_kernels')),
    dict(type='Collect', keys=['img', 'gt_kernels', 'gt_mask'])
]

img_scale_test_icdar2015 = (1333, 736)
test_pipeline_icdar2015 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale_test_icdar2015,  # used by Resize
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# for icdar2017
img_scale_train_icdar2017 = [(3000, 800)]
shrink_ratio_train_icdar2017 = (1.0, 0.5)
target_size_train_icdar2017 = (800, 800)
train_pipeline_icdar2017 = [
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
        img_scale=img_scale_train_icdar2017,
        ratio_range=(0.7, 1.3),
        aspect_ratio_range=(0.9, 1.1),
        multiscale_mode='value',
        keep_ratio=False),
    dict(type='PANetTargets', shrink_ratio=shrink_ratio_train_icdar2017),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomRotateTextDet'),
    dict(
        type='RandomCropInstances',
        target_size=target_size_train_icdar2017,
        instance_key='gt_kernels'),
    dict(type='Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=['gt_kernels', 'gt_mask'],
        visualize=dict(flag=False, boundary_key='gt_kernels')),
    dict(type='Collect', keys=['img', 'gt_kernels', 'gt_mask'])
]

img_scale_test_icdar2017 = (1333, 800)
test_pipeline_icdar2017 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale_test_icdar2017,  # used by Resize
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
