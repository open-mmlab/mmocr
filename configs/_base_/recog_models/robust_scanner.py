label_convertor = dict(
    type='AttnConvertor', dict_type='DICT90', with_unknown=True)

hybrid_decoder = dict(type='SequenceAttentionDecoder')

position_decoder = dict(type='PositionAttentionDecoder')

model = dict(
    type='RobustScanner',
    backbone=dict(type='ResNet31OCR'),
    encoder=dict(
        type='ChannelReductionEncoder',
        in_channels=512,
        out_channels=128,
    ),
    decoder=dict(
        type='RobustScannerDecoder',
        dim_input=512,
        dim_model=128,
        hybrid_decoder=hybrid_decoder,
        position_decoder=position_decoder),
    loss=dict(type='SARLoss'),
    label_convertor=label_convertor,
    max_seq_len=30)
