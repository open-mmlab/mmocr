img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=128,
        max_width=128,
        keep_aspect_ratio=False,
        width_downsample_ratio=0.25),
    dict(
        type='RandomWrapper',
        p=0.5,
        transforms=[
            dict(
                type='OneOfWrapper',
                transforms=[
                    dict(
                        type='RandomRotateTextDet',
                        max_angle=15,
                    ),
                    dict(
                        type='TorchVisionWrapper',
                        op='RandomAffine',
                        degrees=15,
                        translate=(0.3, 0.3),
                        scale=(0.5, 2.),
                        shear=(-45, 45),
                    ),
                    dict(
                        type='TorchVisionWrapper',
                        op='RandomPerspective',
                        distortion_scale=0.5,
                        p=1,
                    ),
                ])
        ],
    ),
    dict(
        type='RandomWrapper',
        p=0.25,
        transforms=[
            dict(type='PyramidRescale'),
            dict(
                type='Albu',
                transforms=[
                    dict(type='GaussNoise', var_limit=(20, 20), p=0.5),
                    dict(type='MotionBlur', blur_limit=6, p=0.5),
                ]),
        ]),
    dict(
        type='RandomWrapper',
        p=0.25,
        transforms=[
            dict(
                type='TorchVisionWrapper',
                op='ColorJitter',
                brightness=0.5,
                saturation=0.5,
                contrast=0.5,
                hue=0.1),
        ]),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'valid_ratio',
            'resize_shape'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiRotateAugOCR',
        rotate_degrees=[0, 90, 270],
        transforms=[
            dict(
                type='ResizeOCR',
                height=32,
                min_width=128,
                max_width=128,
                keep_aspect_ratio=False,
                width_downsample_ratio=0.25),
            dict(type='ToTensorOCR'),
            dict(type='NormalizeOCR', **img_norm_cfg),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'valid_ratio',
                    'resize_shape', 'img_norm_cfg', 'ori_filename'
                ]),
        ])
]
