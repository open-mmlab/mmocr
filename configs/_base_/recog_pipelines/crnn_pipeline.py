img_norm_cfg = dict(mean=[127], std=[127])

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=100,
        max_width=100,
        keep_aspect_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['filename', 'resize_shape', 'text', 'valid_ratio']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='MultiRotateAugOCR',
        rotate_degrees=[0],
        transforms=[
            dict(
                type='ResizeOCR',
                height=32,
                min_width=32,
                max_width=None,
                keep_aspect_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=['filename', 'resize_shape', 'valid_ratio']),
        ])
]
