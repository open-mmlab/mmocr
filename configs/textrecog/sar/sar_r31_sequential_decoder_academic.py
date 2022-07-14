_base_ = [
    'sar.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_adam_step_5e.py',
]

dataset_type = 'OCRDataset'
data_root = 'data/recog/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(type='Resize', scale=(160, 48), keep_ratio=False),
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
        width_divisor=4),
    dict(type='PadToWidth', width=160),
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
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path=None),
        ann_file='train_label.json',
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
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
dictionary = dict(
    type='Dictionary',
    dict_file='dicts/english_digits_symbols.txt',
    with_start=True,
    with_end=True,
    same_start_end=True,
    with_padding=True,
    with_unknown=True)
model = dict(
    type='SARNet',
    backbone=dict(type='ResNet31OCR'),
    encoder=dict(
        type='SAREncoder',
        enc_bi_rnn=False,
        enc_do_rnn=0.1,
        enc_gru=False,
    ),
    decoder=dict(
        type='SequentialSARDecoder',
        enc_bi_rnn=False,
        dec_bi_rnn=False,
        dec_do_rnn=0,
        dec_gru=False,
        pred_dropout=0.1,
        d_k=512,
        pred_concat=True,
        postprocessor=dict(type='AttentionPostprocessor'),
        module_loss=dict(
            type='CEModuleLoss', ignore_first_char=True, reduction='mean')),
    dictionary=dictionary,
    max_seq_len=30)
