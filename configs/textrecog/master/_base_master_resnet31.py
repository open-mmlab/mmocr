dictionary = dict(
    type='Dictionary',
    dict_file='{{ fileDirname }}/../../../dicts/english_digits_symbols.txt',
    with_padding=True,
    with_unknown=True,
    same_start_end=True,
    with_start=True,
    with_end=True)

model = dict(
    type='MASTER',
    backbone=dict(
        type='ResNet',
        in_channels=3,
        stem_channels=[64, 128],
        block_cfgs=dict(
            type='BasicBlock',
            plugins=dict(
                cfg=dict(
                    type='GCAModule',
                    ratio=0.0625,
                    n_head=1,
                    pooling_type='att',
                    is_att_scale=False,
                    fusion_type='channel_add'),
                position='after_conv2')),
        arch_layers=[1, 2, 5, 3],
        arch_channels=[256, 256, 512, 512],
        strides=[1, 1, 1, 1],
        plugins=[
            dict(
                cfg=dict(type='Maxpool2d', kernel_size=2, stride=(2, 2)),
                stages=(True, True, False, False),
                position='before_stage'),
            dict(
                cfg=dict(type='Maxpool2d', kernel_size=(2, 1), stride=(2, 1)),
                stages=(False, False, True, False),
                position='before_stage'),
            dict(
                cfg=dict(
                    type='ConvModule',
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')),
                stages=(True, True, True, True),
                position='after_stage')
        ],
        init_cfg=[
            dict(type='Kaiming', layer='Conv2d'),
            dict(type='Constant', val=1, layer='BatchNorm2d'),
        ]),
    encoder=None,
    decoder=dict(
        type='MasterDecoder',
        d_model=512,
        n_head=8,
        attn_drop=0.,
        ffn_drop=0.,
        d_inner=2048,
        n_layers=3,
        feat_pe_drop=0.2,
        feat_size=6 * 40,
        postprocessor=dict(type='AttentionPostprocessor'),
        module_loss=dict(
            type='CEModuleLoss', reduction='mean', ignore_first_char=True),
        max_seq_len=30,
        dictionary=dictionary),
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5]))

train_pipeline = [
    dict(type='LoadImageFromFile', ignore_empty=True, min_size=2),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='RescaleToHeight',
        height=48,
        min_width=48,
        max_width=160,
        width_divisor=16),
    dict(type='PadToWidth', width=160),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RescaleToHeight',
        height=48,
        min_width=48,
        max_width=160,
        width_divisor=16),
    dict(type='PadToWidth', width=160),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(
                    type='ConditionApply',
                    true_transforms=[
                        dict(
                            type='ImgAugWrapper',
                            args=[dict(cls='Rot90', k=0, keep_size=False)])
                    ],
                    condition="results['img_shape'][1]<results['img_shape'][0]"
                ),
                dict(
                    type='ConditionApply',
                    true_transforms=[
                        dict(
                            type='ImgAugWrapper',
                            args=[dict(cls='Rot90', k=1, keep_size=False)])
                    ],
                    condition="results['img_shape'][1]<results['img_shape'][0]"
                ),
                dict(
                    type='ConditionApply',
                    true_transforms=[
                        dict(
                            type='ImgAugWrapper',
                            args=[dict(cls='Rot90', k=3, keep_size=False)])
                    ],
                    condition="results['img_shape'][1]<results['img_shape'][0]"
                ),
            ],
            [
                dict(
                    type='RescaleToHeight',
                    height=48,
                    min_width=48,
                    max_width=160,
                    width_divisor=16)
            ],
            [dict(type='PadToWidth', width=160)],
            # add loading annotation after ``Resize`` because ground truth
            # does not need to do resize data transform
            [dict(type='LoadOCRAnnotations', with_text=True)],
            [
                dict(
                    type='PackTextRecogInputs',
                    meta_keys=('img_path', 'ori_shape', 'img_shape',
                               'valid_ratio'))
            ]
        ])
]
