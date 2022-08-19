_base_ = [
    '_base_psenet_resnet50_fpnf.py',
    '../../_base_/det_datasets/icdar2015.py',
    '../../_base_/textdet_default_runtime.py',
    '../../_base_/schedules/schedule_adam_step_600e.py',
]

file_client_args = dict(backend='disk')
# dataset settings
ic15_det_train = _base_.ic15_det_train
ic15_det_test = _base_.ic15_det_test

model = _base_.model_quad

train_pipeline_icdar2015 = [
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
    dict(type='ShortScaleAspectJitter', short_size=736, scale_divisor=32),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomRotate', max_angle=10),
    dict(type='TextDetRandomCrop', target_size=(736, 736)),
    dict(type='Pad', size=(736, 736)),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

test_pipeline_icdar2015 = [
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

# pipeline settings
ic15_det_train.pipeline = train_pipeline_icdar2015
ic15_det_test.pipeline = test_pipeline_icdar2015

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=ic15_det_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=ic15_det_test)

test_dataloader = val_dataloader
