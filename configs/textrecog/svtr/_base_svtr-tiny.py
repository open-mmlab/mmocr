dictionary = dict(
    type='Dictionary',
    dict_file='dicts/lower_english_digits.txt',
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
