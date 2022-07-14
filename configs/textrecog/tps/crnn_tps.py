# model
label_convertor = dict(
    type='CTCConvertor', dict_type='DICT36', with_unknown=False, lower=True)

model = dict(
    type='CRNNNet',
    preprocessor=dict(
        type='TPSPreprocessor',
        num_fiducial=20,
        img_size=(32, 100),
        rectified_img_size=(32, 100),
        num_img_channel=1),
    backbone=dict(type='VeryDeepVgg', leaky_relu=False, input_channels=1),
    encoder=None,
    decoder=dict(type='CRNNDecoder', in_channels=512, rnn_flag=True),
    module_loss=dict(type='CTCModuleLoss'),
    label_convertor=label_convertor,
    pretrained=None)
