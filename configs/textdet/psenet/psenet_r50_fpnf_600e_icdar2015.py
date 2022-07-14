_base_ = [
    'psenet_r50_fpnf.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_adam_step_600e.py',
]

model = {{_base_.model_quad}}

train_pipeline_icdar2015 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
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
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(2240, 2240), keep_ratio=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor',
                   'instances'))
]

dataset_type = 'OCRDataset'
data_root = 'data/icdar2015'

train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='instances_training.json',
    data_prefix=dict(img_path='imgs/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline_icdar2015)

test_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='instances_test.json',
    data_prefix=dict(img_path='imgs/'),
    test_mode=True,
    pipeline=test_pipeline_icdar2015)

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)
test_dataloader = val_dataloader

val_evaluator = dict(type='HmeanIOUMetric')
test_evaluator = val_evaluator

visualizer = dict(type='TextDetLocalVisualizer', name='visualizer')
