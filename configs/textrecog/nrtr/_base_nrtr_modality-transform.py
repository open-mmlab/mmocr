file_client_args = dict(backend='disk')

dictionary = dict(
    type='Dictionary',
    dict_file='{{ fileDirname }}/../../../dicts/english_digits_symbols.txt',
    with_padding=True,
    with_unknown=True,
    same_start_end=True,
    with_start=True,
    with_end=True)

model = dict(
    type='NRTR',
    backbone=dict(type='NRTRModalityTransform'),
    encoder=dict(type='NRTREncoder', n_layers=12),
    decoder=dict(
        type='NRTRDecoder',
        module_loss=dict(
            type='CEModuleLoss', ignore_first_char=True, flatten=True),
        postprocessor=dict(type='AttentionPostprocessor'),
        dictionary=dictionary,
        max_seq_len=30),
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=file_client_args,
        ignore_empty=True,
        min_size=2),
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
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]
