img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=32,
        max_width=160,
        keep_aspect_ratio=True,
        width_downsample_ratio=0.25),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'resize_shape', 'text', 'valid_ratio'
        ]),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiRotateAugOCR',
        rotate_degrees=[0],
        transforms=[
            dict(
                type='ResizeOCR',
                height=32,
                min_width=32,
                max_width=160,
                keep_aspect_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'resize_shape', 'valid_ratio'
                ])
        ])
]
