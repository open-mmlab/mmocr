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
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        norm_eval=False,
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
        module_loss=None,
        postprocessor=dict(type='SPTSPostprocessor', num_bins=num_bins)))

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=file_client_args,
        color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(1000, 1824), keep_ratio=True),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
        with_text=True),
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
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
        with_text=True),
    dict(type='RemoveIgnored'),
    dict(type='RandomCrop', min_side_ratio=0.1),
    dict(
        type='RandomRotate',
        max_angle=30,
        pad_with_fixed_color=True,
        use_canvas=True),
    dict(
        type='RandomChoiceResize',
        scales=[(980, 2900), (1044, 2900), (1108, 2900), (1172, 2900),
                (1236, 2900), (1300, 2900), (1364, 2900), (1428, 2900),
                (1492, 2900)],
        keep_ratio=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
