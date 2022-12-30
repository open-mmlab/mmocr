_base_ = [
    'svtr-tiny_20e_st_mj.py',
]

model = dict(
    preprocessor=dict(output_image_size=(48, 160), ),
    encoder=dict(
        img_size=[48, 160],
        max_seq_len=40,
        out_channels=384,
        embed_dims=[192, 256, 512],
        depth=[3, 9, 9],
        num_heads=[6, 8, 16],
        mixer_types=['Local'] * 10 + ['Global'] * 11),
    decoder=dict(in_channels=256))

train_dataloader = dict(batch_size=128, )

optim_wrapper = dict(optimizer=dict(lr=2.5 / (10**4)))
