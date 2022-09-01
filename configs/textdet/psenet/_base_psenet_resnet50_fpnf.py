file_client_args = dict(backend='disk')

model = dict(
    type='PSENet',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        norm_eval=True,
        style='caffe'),
    neck=dict(
        type='FPNF',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        fusion_type='concat'),
    det_head=dict(
        type='PSEHead',
        in_channels=[256],
        hidden_dim=256,
        out_channel=7,
        module_loss=dict(type='PSEModuleLoss'),
        postprocessor=dict(type='PSEPostprocessor', text_repr_type='poly')),
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
        with_label=True),
    dict(
        type='TorchVisionWrapper',
        op='ColorJitter',
        brightness=32.0 / 255,
        saturation=0.5),
    dict(type='FixInvalidPolygon'),
    dict(type='ShortScaleAspectJitter', short_size=736, scale_divisor=32),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomRotate', max_angle=10),
    dict(type='TextDetRandomCrop', target_size=(736, 736)),
    dict(type='Pad', size=(736, 736)),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=file_client_args,
        color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(2240, 2240), keep_ratio=True),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
