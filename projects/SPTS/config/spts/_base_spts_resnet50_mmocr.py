_base_ = '_base_spts_resnet50.py'

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='RescaleToShortSide',
        short_side_lens=[1000],
        long_side_bound=1824),
    dict(
        type='LoadOCRAnnotations',
        with_bbox=True,
        with_label=True,
        with_polygon=True,
        with_text=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotations',
        with_bbox=True,
        with_label=True,
        with_polygon=True,
        with_text=True),
    dict(type='FixInvalidPolygon'),
    dict(type='RemoveIgnored'),
    dict(type='RandomCrop', min_side_ratio=0.5),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='RandomRotate',
                max_angle=30,
                pad_with_fixed_color=True,
                use_canvas=True)
        ],
        prob=0.3),
    dict(type='FixInvalidPolygon'),
    dict(
        type='RandomChoiceResize',
        scales=[(640, 1600), (672, 1600), (704, 1600), (736, 1600),
                (768, 1600), (800, 1600), (832, 1600), (864, 1600),
                (896, 1600)],
        keep_ratio=True),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='TorchVisionWrapper',
                op='ColorJitter',
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.5)
        ],
        prob=0.5),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
