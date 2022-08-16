st_data_root = 'data/rec/SynthText/'

st_rec_train = dict(
    type='OCRDataset',
    data_root=st_data_root,
    data_prefix=dict(img_path='synthtext/SynthText_patch_horizontal'),
    ann_file='train_labels.json',
    test_mode=False,
    pipeline=None)

st_an_rec_train = dict(
    type='OCRDataset',
    data_root=st_data_root,
    data_prefix=dict(img_path='synthtext/SynthText_patch_horizontal'),
    ann_file='alphanumeric_train_labels.json',
    test_mode=False,
    pipeline=None)

st_sub_rec_train = dict(
    type='OCRDataset',
    data_root=st_data_root,
    data_prefix=dict(img_path='synthtext/SynthText_patch_horizontal'),
    ann_file='subset_train_labels.json',
    test_mode=False,
    pipeline=None)
