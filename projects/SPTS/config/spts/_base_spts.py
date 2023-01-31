file_client_args = dict(backend='disk')

dictionary = dict(
    type='SPTSDictionary',
    dict_file='{{ fileDirname }}/../../dicts/spts.txt',
    with_start=True,
    with_end=True,
    with_seq_end=True,
    same_start_end=False,
    with_padding=True,
    with_unknown=True,
    unknown_token=None,
)

num_bins = 1000

model = dict(
    type='SPTS',
    data_preprocessor=dict(
        type='TextDetDataPreprocessor',
        # mean=[123.675, 116.28, 103.53][::-1],
        # std=[1, 1, 1],
        mean=[0, 0, 0],
        std=[255, 255, 255],
        bgr_to_rgb=True),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),  # freeze w & b
        norm_eval=True,  # freeze running mean and var
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'),
    encoder=dict(
        type='SPTSEncoder',
        d_backbone=2048,
        d_model=256,
    ),
    decoder=dict(
        type='SPTSDecoder',
        dictionary=dictionary,
        num_bins=num_bins,
        d_model=256,
        dropout=0.1,
        max_num_text=60,
        module_loss=dict(
            type='SPTSModuleLoss', num_bins=num_bins, ignore_first_char=True),
        postprocessor=dict(type='SPTSPostprocessor', num_bins=num_bins)))

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=file_client_args,
        color_type='color_ignore_orientation'),
    # dict(type='Resize', scale=(1000, 1824), keep_ratio=True),
    dict(
        type='RescaleToShortSide',
        short_side_lens=[1000],
        long_side_bound=1824),
    dict(
        type='LoadOCRAnnotationsWithBezier',
        with_bbox=True,
        with_label=True,
        with_bezier=True,
        with_text=True),
    dict(type='Bezier2Polygon'),
    dict(type='ConvertText', dictionary=dictionary),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=file_client_args,
        color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotationsWithBezier',
        with_bbox=True,
        with_label=True,
        with_bezier=True,
        with_text=True),
    dict(type='Bezier2Polygon'),
    dict(type='FixInvalidPolygon'),
    dict(type='ConvertText', dictionary=dictionary),
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
    # dict(type='Polygon2Bezier'),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
