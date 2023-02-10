_base_ = '_base_sdmgr_novisual.py'

model = dict(
    backbone=dict(type='UNet', base_channels=16),
    roi_extractor=dict(
        type='mmdet.SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', output_size=7),
        featmap_strides=[1]),
    data_preprocessor=dict(
        type='ImgDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadKIEAnnotations'),
    dict(type='Resize', scale=(1024, 512), keep_ratio=True),
    dict(type='PackKIEInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadKIEAnnotations'),
    dict(type='Resize', scale=(1024, 512), keep_ratio=True),
    dict(type='PackKIEInputs', meta_keys=('img_path', )),
]
