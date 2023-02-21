dictionary = dict(
    type='Dictionary',
    dict_file='{{ fileDirname }}/../../../dicts/lower_english_digits.txt',
    with_padding=True,
    with_unknown=True,
)

model = dict(
    type='SVTR',
    preprocessor=dict(
        type='STN',
        in_channels=3,
        resized_image_size=(32, 64),
        output_image_size=(32, 100),
        num_control_points=20,
        margins=[0.05, 0.05]),
    encoder=dict(
        type='SVTREncoder',
        img_size=[32, 100],
        in_channels=3,
        out_channels=192,
        embed_dims=[64, 128, 256],
        depth=[3, 6, 3],
        num_heads=[2, 4, 8],
        mixer_types=['Local'] * 6 + ['Global'] * 6,
        window_size=[[7, 11], [7, 11], [7, 11]],
        merging_types='Conv',
        prenorm=False,
        max_seq_len=25),
    decoder=dict(
        type='SVTRDecoder',
        in_channels=192,
        module_loss=dict(
            type='CTCModuleLoss', letter_case='lower', zero_infinity=True),
        postprocessor=dict(type='CTCPostProcessor'),
        dictionary=dictionary),
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor', mean=[127.5], std=[127.5]))

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=file_client_args,
        ignore_empty=True,
        min_size=5),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='RandomApply',
        prob=0.4,
        transforms=[
            dict(type='TextRecogGeneralAug', ),
        ],
    ),
    dict(
        type='RandomApply',
        prob=0.4,
        transforms=[
            dict(type='CropHeight', ),
        ],
    ),
    dict(
        type='ConditionApply',
        condition='min(results["img_shape"])>10',
        true_transforms=dict(
            type='RandomApply',
            prob=0.4,
            transforms=[
                dict(
                    type='TorchVisionWrapper',
                    op='GaussianBlur',
                    kernel_size=5,
                    sigma=1,
                ),
            ],
        )),
    dict(
        type='RandomApply',
        prob=0.4,
        transforms=[
            dict(
                type='TorchVisionWrapper',
                op='ColorJitter',
                brightness=0.5,
                saturation=0.5,
                contrast=0.5,
                hue=0.1),
        ]),
    dict(
        type='RandomApply',
        prob=0.4,
        transforms=[
            dict(type='ImageContentJitter', ),
        ],
    ),
    dict(
        type='RandomApply',
        prob=0.4,
        transforms=[
            dict(
                type='ImgAugWrapper',
                args=[dict(cls='AdditiveGaussianNoise', scale=0.1**0.5)]),
        ],
    ),
    dict(
        type='RandomApply',
        prob=0.4,
        transforms=[
            dict(type='ReversePixels', ),
        ],
    ),
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
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
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
                                       'valid_ratio'))
                    ]])
]
