dictionary = dict(
    type='Dictionary',
    dict_file='dicts/lower_english_digits.txt',
    with_padding=True)

model = dict(
    type='CRNNNet',
    preprocessor=None,
    backbone=dict(type='VeryDeepVgg', leaky_relu=False, input_channels=1),
    encoder=None,
    decoder=dict(
        type='CRNNDecoder',
        in_channels=512,
        rnn_flag=True,
        loss=dict(type='CTCLoss', letter_case='lower'),
        postprocessor=dict(type='CTCPostProcessor')),
    dictionary=dictionary,
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor', mean=[127], std=[127]))
