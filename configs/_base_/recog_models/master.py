label_convertor = dict(
    type='MasterConvertor', dict_type='DICT90', with_unknown=True)

if label_convertor['dict_type'] == 'DICT90':
    PAD = 92
else:
    raise ValueError

model = dict(
    type='MASTER',
    backbone=dict(
        type='ResNetMASTER',
        input_dim=3,
        gcb_config=dict(
            ratio=0.0625,
            headers=1,
            att_scale=False,
            fusion_type="channel_add",
            layers=[False, True, True, True],
        ),
        layers=[1, 2, 5, 3]),
    encoder=dict(
        type='PositionalEncoder',
        d_model=512,
        dropout=0.2,
        max_len=5000),
    decoder=dict(
        type='MasterDecoder',
        N=3,
        decoder=dict(
            self_attn=dict(
                headers=8,
                d_model=512,
                dropout=0.),
            src_attn=dict(
                headers=8,
                d_model=512,
                dropout=0.),
            feed_forward=dict(
                d_model=512,
                d_ff=2024,
                dropout=0.),
            size=512,
            dropout=0.),
        d_model=512),
    loss=dict(type='MASTERTFLoss', ignore_index=PAD, reduction='mean'),
    label_convertor=label_convertor,
    max_seq_len=30)
