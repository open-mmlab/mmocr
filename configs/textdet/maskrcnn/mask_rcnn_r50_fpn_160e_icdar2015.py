_base_ = [
    'ocr_mask_rcnn_r50_fpn_ohem_poly.py',
    '../../_base_/det_datasets/toy_data.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_sgd_160e.py',
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
        type='TorchVisionWrapper',
        op='ColorJitter',
        brightness=32.0 / 255,
        saturation=0.5,
        contrast=0.5),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(1.0, 4.125),
        resize_type='mmocr.Resize',
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='TextDetRandomCrop', target_size=(640, 640)),
    dict(type='MMOCR2MMDet', poly2mask=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'flip',
                   'scale_factor', 'flip_direction'))
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=file_client_args,
        color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(1920, 1920), keep_ratio=True),
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
