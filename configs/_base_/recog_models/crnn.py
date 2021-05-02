label_convertor = dict(
    type='CTCConvertor', dict_type='DICT90', with_unknown=False)

model = dict(
    type='CRNNNet',
    preprocessor=None,
    backbone=dict(type='VeryDeepVgg', leaky_relu=False),
    encoder=None,
    decoder=dict(type='CRNNDecoder', in_channels=512, rnn_flag=True),
    loss=dict(type='CTCLoss', flatten=False),
    label_convertor=label_convertor)
