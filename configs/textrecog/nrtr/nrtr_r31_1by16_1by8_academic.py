_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_adam_step_6e.py'
]

optimizer = dict(type='Adam', lr=3e-4)
default_hooks = dict(logger=dict(type='LoggerHook', interval=50))

dictionary = dict(
    type='Dictionary',
    dict_file='dicts/english_digits_symbols.txt',
    with_padding=True,
    with_unknown=True,
    same_start_end=True,
    with_start=True,
    with_end=True)

model = dict(
    type='NRTR',
    backbone=dict(
        type='ResNet31OCR',
        layers=[1, 2, 5, 3],
        channels=[32, 64, 128, 256, 512, 512],
        stage4_pool_cfg=dict(kernel_size=(2, 1), stride=(2, 1)),
        last_stage_pool=True),
    encoder=dict(type='NRTREncoder'),
    decoder=dict(
        type='NRTRDecoder',
        module_loss=dict(
            type='CEModuleLoss', ignore_first_char=True, flatten=True),
        postprocessor=dict(type='AttentionPostprocessor')),
    dictionary=dictionary,
    max_seq_len=30,
    preprocess_cfg=dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]))

# dataset settings
dataset_type = 'OCRDataset'
data_root = 'data/recog/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='RescaleToHeight',
        height=32,
        min_width=32,
        max_width=160,
        width_divisor=4),
    dict(type='PadToWidth', width=160),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='RescaleToHeight',
        height=32,
        min_width=32,
        max_width=160,
        width_divisor=16),
    dict(type='PadToWidth', width=160),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio',
                   'instances'))
]

train_dataloader = dict(
    batch_size=256,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path=None),
        ann_file='train_label.json',
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=128,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path=None),
        ann_file='test_label.json',
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type='WordMetric', mode=['exact', 'ignore_case',
                                 'ignore_case_symbol']),
    dict(type='CharMetric')
]

test_evaluator = val_evaluator
visualizer = dict(type='TextRecogLocalVisualizer', name='visualizer')
