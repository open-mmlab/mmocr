# training schedule for 1x
_base_ = [
    'crnn.py',
    '../../_base_/default_runtime.py',
    '../../_base_/recog_datasets/MJ_train.py',
    '../../_base_/recog_datasets/academic_test.py',
    '../../_base_/schedules/schedule_adadelta_5e.py',
]

# dataset settings
train_list = {{_base_.train_list}}

file_client_args = dict(backend='disk')
default_hooks = dict(logger=dict(type='LoggerHook', interval=50), )

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        color_type='grayscale',
        file_client_args=file_client_args),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(type='Resize', scale=(100, 32), keep_ratio=False),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        color_type='grayscale',
        file_client_args=file_client_args),
    dict(
        type='RescaleToHeight',
        height=32,
        min_width=32,
        max_width=None,
        width_divisor=16),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio',
                   'instances'))
]

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset', datasets=train_list, pipeline=train_pipeline))

test_cfg = dict(type='MultiTestLoop')
val_cfg = dict(type='MultiValLoop')
val_dataloader = _base_.val_dataloader
test_dataloader = _base_.test_dataloader
for dataloader in test_dataloader:
    dataloader['dataset']['pipeline'] = test_pipeline
for dataloader in val_dataloader:
    dataloader['dataset']['pipeline'] = test_pipeline

visualizer = dict(type='TextRecogLocalVisualizer', name='visualizer')
