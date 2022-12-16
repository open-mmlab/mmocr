file_client_args = dict(backend='disk')

model = dict(
    type='FCENet',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True,
        act_cfg=None),
    det_head=dict(
        type='FCEHead',
        in_channels=256,
        fourier_degree=5,
        module_loss=dict(type='FCEModuleLoss', num_sample=50),
        postprocessor=dict(
            type='FCEPostprocessor',
            scales=(8, 16, 32),
            text_repr_type='quad',
            num_reconstr_points=50,
            alpha=1.2,
            beta=1.0,
            score_thr=0.3)),
    data_preprocessor=dict(
        type='TextDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32))

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
    ),
    dict(type='FixInvalidPolygon', min_poly_points=4),
    dict(type='RemoveIgnored'),
    dict(
        type='RandomResize',
        scale=(800, 800),
        ratio_range=(0.75, 2.5),
        keep_ratio=True),
    dict(
        type='TextDetRandomCropFlip',
        crop_ratio=0.5,
        iter_num=1,
        min_area_ratio=0.2),
    dict(
        type='FixInvalidPolygon',
        min_poly_points=4,
        prompt='TextDetRandomCropFlipe'),
    dict(
        type='RandomApply',
        transforms=[dict(type='RandomCrop', min_side_ratio=0.3)],
        prob=0.8),
    dict(type='FixInvalidPolygon', min_poly_points=4, prompt='RandomCrop'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='RandomRotate',
                max_angle=30,
                pad_with_fixed_color=False,
                use_canvas=True)
        ],
        prob=0.5),
    dict(type='FixInvalidPolygon', min_poly_points=4, prompt='RandomRotate'),
    dict(
        type='RandomChoice',
        transforms=[[
            dict(type='Resize', scale=800, keep_ratio=True),
            dict(type='SourceImagePad', target_scale=800)
        ],
                    dict(type='Resize', scale=800, keep_ratio=False)],
        prob=[0.6, 0.4]),
    dict(type='FixInvalidPolygon', min_poly_points=4, prompt='RandomChoice'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='FixInvalidPolygon', min_poly_points=4, prompt='RandomFlip'),
    dict(
        type='TorchVisionWrapper',
        op='ColorJitter',
        brightness=32.0 / 255,
        saturation=0.5,
        contrast=0.5),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=file_client_args,
        color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(2260, 2260), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
