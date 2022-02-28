num_chars = 37
max_seq_len = 26

label_convertor = dict(
    type='ABIConvertor',
    dict_type='DICT36',
    with_unknown=False,
    with_padding=False,
    lower=True,
)

model = dict(
    type='ABINet',
    backbone=dict(type='ResNetABI'),
    encoder=dict(
        type='ABIVisionModel',
        encoder=dict(
            type='TransformerEncoder',
            n_layers=3,
            n_head=8,
            d_model=512,
            d_inner=2048,
            dropout=0.1,
            max_len=8 * 32,
        ),
        decoder=dict(
            type='ABIVisionDecoder',
            in_channels=512,
            num_channels=64,
            attn_height=8,
            attn_width=32,
            attn_mode='nearest',
            use_result='feature',
            num_chars=num_chars,
            max_seq_len=max_seq_len,
            init_cfg=dict(type='Xavier', layer='Conv2d')),
    ),
    decoder=dict(
        type='ABILanguageDecoder',
        d_model=512,
        n_head=8,
        d_inner=2048,
        n_layers=4,
        dropout=0.1,
        detach_tokens=True,
        use_self_attn=False,
        pad_idx=num_chars - 1,
        num_chars=num_chars,
        max_seq_len=max_seq_len,
        init_cfg=None),
    fuser=dict(
        type='ABIFuser',
        d_model=512,
        num_chars=num_chars,
        init_cfg=None,
        max_seq_len=max_seq_len,
    ),
    loss=dict(
        type='ABILoss',
        enc_weight=1.0,
        dec_weight=1.0,
        fusion_weight=1.0,
        num_classes=num_chars),
    label_convertor=label_convertor,
    max_seq_len=max_seq_len,
    iter_size=3)
