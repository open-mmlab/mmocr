img_norm_cfg = dict(mean=[0.5], std=[0.5])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=100,
        max_width=100,
        keep_aspect_ratio=False),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
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
        type='ResizeOCR',
        height=32,
        min_width=32,
        max_width=100,
        keep_aspect_ratio=False),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'resize_shape', 'valid_ratio',
            'img_norm_cfg', 'ori_filename', 'img_shape'
        ]),
]
