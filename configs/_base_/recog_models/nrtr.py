label_convertor = dict(
    type='AttnConvertor', dict_type='DICT36', with_unknown=True, lower=True)

model = dict(
    type='TransformerNet',
    backbone=dict(type='NRTRModalityTransform'),
    encoder=dict(type='TFEncoder'),
    decoder=dict(type='TFDecoder'),
    loss=dict(type='TFLoss'),
    label_convertor=label_convertor,
    max_seq_len=40)
