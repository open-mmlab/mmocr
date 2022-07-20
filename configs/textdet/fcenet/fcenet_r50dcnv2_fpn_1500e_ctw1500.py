_base_ = [
    'fcenet_r50dcnv2_fpn.py',
    '../../_base_/det_datasets/ctw1500.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_sgd_1500e.py',
]

# dataset settings
train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}
file_client_args = dict(backend='disk')
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=20),
    logger=dict(type='LoggerHook', interval=20))

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
        type='RandomApply',
        transforms=[dict(type='RandomCrop', min_side_ratio=0.3)],
        prob=0.8),
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
    dict(
        type='RandomChoice',
        transforms=[[
            dict(type='Resize', scale=800, keep_ratio=True),
            dict(type='SourceImagePad', target_scale=800)
        ],
                    dict(type='Resize', scale=800, keep_ratio=False)],
        prob=[0.6, 0.4]),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
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
    dict(type='Resize', scale=(1080, 736), keep_ratio=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor',
                   'instances'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset', datasets=train_list, pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset', datasets=test_list, pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='HmeanIOUMetric')
test_evaluator = val_evaluator

visualizer = dict(
    type='TextDetLocalVisualizer', name='visualizer', save_dir='imgs')
