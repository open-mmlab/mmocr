max_seq_len = 40

label_convertor = dict(
    type='AttnConvertor',
    dict_type='DICT36',
    with_unknown=True,
    lower=True,
    max_seq_len=max_seq_len)

model = dict(
    type='NRTR',
    backbone=dict(type='NRTRModalityTransform'),
    encoder=dict(type='NRTREncoder', n_layers=12),
    decoder=dict(type='NRTRDecoder'),
    loss=dict(type='TFLoss'),
    label_convertor=label_convertor,
    max_seq_len=max_seq_len)
