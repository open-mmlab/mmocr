_base_ = '_base_abinet-vision.py'

model = dict(
    decoder=dict(
        d_model=512,
        num_iters=3,
        language_decoder=dict(
            type='ABILanguageDecoder',
            d_model=512,
            n_head=8,
            d_inner=2048,
            n_layers=4,
            dropout=0.1,
            detach_tokens=True,
            use_self_attn=False,
        )), )
