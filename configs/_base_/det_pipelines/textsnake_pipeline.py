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
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='TextSnakeTargets'),
    dict(type='Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=[
            'gt_text_mask', 'gt_center_region_mask', 'gt_mask',
            'gt_radius_map', 'gt_sin_map', 'gt_cos_map'
        ],
        visualize=dict(flag=False, boundary_key='gt_text_mask')),
    dict(
        type='Collect',
        keys=[
            'img', 'gt_text_mask', 'gt_center_region_mask', 'gt_mask',
            'gt_radius_map', 'gt_sin_map', 'gt_cos_map'
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 736),  # used by Resize
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
