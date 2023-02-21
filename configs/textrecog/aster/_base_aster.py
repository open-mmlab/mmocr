file_client_args = dict(backend='disk')

dictionary = dict(
    type='Dictionary',
    dict_file='{{ fileDirname }}/../../../dicts/english_digits_symbols.txt',
    with_padding=True,
    with_unknown=True,
    same_start_end=True,
    with_start=True,
    with_end=True)

model = dict(
    type='ASTER',
    preprocessor=dict(
        type='STN',
        in_channels=3,
        resized_image_size=(32, 64),
        output_image_size=(32, 100),
        num_control_points=20),
    backbone=dict(
        type='ResNet',
        in_channels=3,
        stem_channels=[32],
        block_cfgs=dict(type='BasicBlock', use_conv1x1='True'),
        arch_layers=[3, 4, 6, 6, 3],
        arch_channels=[32, 64, 128, 256, 512],
        strides=[(2, 2), (2, 2), (2, 1), (2, 1), (2, 1)],
        init_cfg=[
            dict(type='Kaiming', layer='Conv2d'),
            dict(type='Constant', val=1, layer='BatchNorm2d'),
        ]),
    encoder=dict(type='ASTEREncoder', in_channels=512),
    decoder=dict(
        type='ASTERDecoder',
        max_seq_len=25,
        in_channels=512,
        emb_dims=512,
        attn_dims=512,
        hidden_size=512,
        postprocessor=dict(type='AttentionPostprocessor'),
        module_loss=dict(
            type='CEModuleLoss', flatten=True, ignore_first_char=True),
        dictionary=dictionary,
    ),
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5]))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=file_client_args,
        ignore_empty=True,
        min_size=5),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(type='Resize', scale=(256, 64)),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(256, 64)),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio',
                   'instances'))
]

tta_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(
                type='ConditionApply',
                true_transforms=[
                    dict(
                        type='ImgAugWrapper',
                        args=[dict(cls='Rot90', k=0, keep_size=False)])
                ],
                condition="results['img_shape'][1]<results['img_shape'][0]"),
            dict(
                type='ConditionApply',
                true_transforms=[
                    dict(
                        type='ImgAugWrapper',
                        args=[dict(cls='Rot90', k=1, keep_size=False)])
                ],
                condition="results['img_shape'][1]<results['img_shape'][0]"),
            dict(
                type='ConditionApply',
                true_transforms=[
                    dict(
                        type='ImgAugWrapper',
                        args=[dict(cls='Rot90', k=3, keep_size=False)])
                ],
                condition="results['img_shape'][1]<results['img_shape'][0]"),
        ], [dict(type='Resize', scale=(256, 64))],
                    [dict(type='LoadOCRAnnotations', with_text=True)],
                    [
                        dict(
                            type='PackTextRecogInputs',
                            meta_keys=('img_path', 'ori_shape', 'img_shape',
                                       'valid_ratio', 'instances'))
                    ]])
]
