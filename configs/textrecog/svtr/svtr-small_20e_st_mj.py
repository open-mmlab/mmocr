_base_ = [
    'svtr-tiny_20e_st_mj.py',
]

model = dict(
    backbone=dict(
        embed_dims=[96, 192, 256],
        depth=[3, 6, 6],
        num_heads=[3, 6, 8],
        mixer_types=['Local'] * 8 + ['Global'] * 7))
