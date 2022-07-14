dictionary = dict(
    type='Dictionary',
    dict_file='dicts/lower_english_digits.txt',
    with_start=True,
    with_end=True,
    same_start_end=True,
    with_padding=False,
    with_unknown=False)

model = dict(
    type='ABINet',
    backbone=dict(type='ResNetABI'),
    encoder=dict(
        type='ABIEncoder',
        n_layers=3,
        n_head=8,
        d_model=512,
        d_inner=2048,
        dropout=0.1,
        max_len=8 * 32,
    ),
    decoder=dict(
        type='ABIFuser',
        vision_decoder=dict(
            type='ABIVisionDecoder',
            in_channels=512,
            num_channels=64,
            attn_height=8,
            attn_width=32,
            attn_mode='nearest',
            init_cfg=dict(type='Xavier', layer='Conv2d')),
        module_loss=dict(type='ABIModuleLoss', letter_case='lower'),
        postprocessor=dict(type='AttentionPostprocessor'),
        dictionary=dictionary,
        max_seq_len=26,
    ),
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]))
