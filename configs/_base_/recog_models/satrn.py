max_seq_len = 40

label_convertor = dict(
    type='AttnConvertor',
    dict_type='DICT36',
    with_unknown=True,
    lower=True,
    max_seq_len=max_seq_len)

model = dict(
    type='SATRN',
    backbone=dict(type='ShallowCNN'),
    encoder=dict(type='SatrnEncoder'),
    decoder=dict(type='TFDecoder'),
    loss=dict(type='TFLoss'),
    label_convertor=label_convertor,
    max_seq_len=max_seq_len)
