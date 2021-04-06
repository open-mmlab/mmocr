label_convertor = dict(
    type='AttnConvertor', dict_type='DICT90', with_unknown=False)

model = dict(
    type='TransformerNet',
    backbone=dict(type='ResNet31OCR'),
    encoder=dict(type='TFEncoder'),
    decoder=dict(type='TFDecoder'),
    loss=dict(type='TFLoss'),
    label_convertor=label_convertor,
    max_seq_len=40)
