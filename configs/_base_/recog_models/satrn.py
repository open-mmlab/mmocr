dictionary = dict(
    type='Dictionary',
    dict_file='dicts/english_digits_symbols.txt',
    with_padding=True,
    with_unknown=True,
    same_start_end=True,
    with_start=True,
    with_end=True)

model = dict(
    type='SATRN',
    backbone=dict(type='ShallowCNN'),
    encoder=dict(type='SATRNEncoder'),
    decoder=dict(
        type='NRTRDecoder',
        module_loss=dict(type='CEModuleLoss'),
        dictionary=dictionary,
        max_seq_len=40),
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]))
