img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

gt_label_convertor = dict(
    type='SegConvertor', dict_type='DICT36', with_unknown=True, lower=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomPaddingOCR',
        max_ratio=[0.15, 0.2, 0.15, 0.2],
        box_type='char_quads'),
    dict(type='OpencvToPil'),
    dict(
        type='RandomRotateImageBox',
        min_angle=-17,
        max_angle=17,
        box_type='char_quads'),
    dict(type='PilToOpencv'),
    dict(
        type='ResizeOCR',
        height=64,
        min_width=64,
        max_width=512,
        keep_aspect_ratio=True),
    dict(
        type='OCRSegTargets',
        label_convertor=gt_label_convertor,
        box_type='char_quads'),
    dict(type='RandomRotateTextDet', rotate_ratio=0.5, max_angle=15),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='ToTensorOCR'),
    dict(type='FancyPCA'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='CustomFormatBundle',
        keys=['gt_kernels'],
        visualize=dict(flag=False, boundary_key=None),
        call_super=False),
    dict(
        type='Collect',
        keys=['img', 'gt_kernels'],
        meta_keys=['filename', 'ori_shape', 'resize_shape'])
]

test_img_norm_cfg = dict(
    mean=[x * 255 for x in img_norm_cfg['mean']],
    std=[x * 255 for x in img_norm_cfg['std']])

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=64,
        min_width=64,
        max_width=None,
        keep_aspect_ratio=True),
    dict(type='Normalize', **test_img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['filename', 'ori_shape', 'resize_shape'])
]
