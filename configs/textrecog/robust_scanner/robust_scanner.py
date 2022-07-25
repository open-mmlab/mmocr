dictionary = dict(
    type='Dictionary',
    dict_file='dicts/english_digits_symbols.txt',
    with_start=True,
    with_end=True,
    same_start_end=True,
    with_padding=True,
    with_unknown=True)

model = dict(
    type='RobustScanner',
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor',
        mean=[127, 127, 127],
        std=[127, 127, 127]),
    backbone=dict(type='ResNet31OCR'),
    encoder=dict(
        type='ChannelReductionEncoder', in_channels=512, out_channels=128),
    decoder=dict(
        type='RobustScannerFuser',
        hybrid_decoder=dict(
            type='SequenceAttentionDecoder', dim_input=512, dim_model=128),
        position_decoder=dict(
            type='PositionAttentionDecoder', dim_input=512, dim_model=128),
        in_channels=[512, 512],
        postprocessor=dict(type='AttentionPostprocessor'),
        module_loss=dict(
            type='CEModuleLoss', ignore_first_char=True, reduction='mean'),
        dictionary=dictionary,
        max_seq_len=30))
