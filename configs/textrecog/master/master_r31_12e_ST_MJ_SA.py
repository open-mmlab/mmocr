_base_ = [
    'master.py',
    '../../_base_/recog_datasets/ST_SA_MJ_train.py',
    '../../_base_/recog_datasets/academic_test.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_adam_step_12e.py',
]

# dataset settings
train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}
file_client_args = dict(backend='disk')
default_hooks = dict(logger=dict(type='LoggerHook', interval=50), )

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='RescaleToHeight',
        height=48,
        min_width=48,
        max_width=160,
        width_divisor=16),
    dict(type='PadToWidth', width=160),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='RescaleToHeight',
        height=48,
        min_width=48,
        max_width=160,
        width_divisor=16),
    dict(type='PadToWidth', width=160),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio',
                   'instances'))
]

train_dataloader = dict(
    batch_size=512,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset', datasets=train_list, pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=128,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset', datasets=test_list, pipeline=train_pipeline))
test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type='WordMetric', mode=['exact', 'ignore_case',
                                 'ignore_case_symbol']),
    dict(type='CharMetric')
]
test_evaluator = val_evaluator
visualizer = dict(type='TextRecogLocalVisualizer', name='visualizer')
