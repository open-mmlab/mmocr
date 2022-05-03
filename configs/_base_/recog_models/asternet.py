# model
label_convertor = dict(
    type='AttnConvertor', dict_type='DICT36', with_unknown=True, lower=True)

model = dict(
    type='ASTERNet',
    preprocessor=dict(
        type='TPSPreprocessor',
        img_size=(32, 100),
        rectified_img_size=(32, 100),
        num_img_channel=3),
    backbone=dict(
        type='ResNetABI',
        arch_settings=[3, 4, 6, 6, 3],
        strides=[2, 2, [2, 1], [2, 1], [2, 1]],
        init_cfg=dict(type='Xavier', layer='Conv2d')),
    encoder=dict(
        type='ASTEREncoder', in_channels=512, num_classes=512, with_lstm=True),
    decoder=dict(
        type='ASTERDecoder',
        in_channels=512,
        num_classes=512,
        s_Dim=512,
        Atten_Dim=512,
        max_seq_len=40),
    loss=dict(type='CELoss'),
    label_convertor=label_convertor,
    pretrained=None)
