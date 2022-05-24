label_convertor = dict(
    type='AttnConvertor', dict_type='DICT90', with_unknown=True)

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
        feat_size=6 * 40),
    loss=dict(type='TFLoss', reduction='mean'),
    label_convertor=label_convertor,
    max_seq_len=30)
