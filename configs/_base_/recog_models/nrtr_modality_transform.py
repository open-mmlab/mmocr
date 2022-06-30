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
    backbone=dict(type='NRTRModalityTransform'),
    encoder=dict(type='NRTREncoder', n_layers=12),
    decoder=dict(
        type='NRTRDecoder',
        loss_module=dict(type='CELoss', ignore_first_char=True, flatten=True),
        postprocessor=dict(type='AttentionPostprocessor')),
    dictionary=dictionary,
    max_seq_len=30,
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]))
