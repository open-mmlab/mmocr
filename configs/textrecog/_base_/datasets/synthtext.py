synthtext_textrecog_data_root = 'data/rec/SynthText/'

synthtext_textrecog_train = dict(
    type='OCRDataset',
    data_root=synthtext_textrecog_data_root,
    data_prefix=dict(img_path='synthtext/SynthText_patch_horizontal'),
    ann_file='train_labels.json',
    test_mode=False,
    pipeline=None)

synthtext_an_textrecog_train = dict(
    type='OCRDataset',
    data_root=synthtext_textrecog_data_root,
    data_prefix=dict(img_path='synthtext/SynthText_patch_horizontal'),
    ann_file='alphanumeric_train_labels.json',
    test_mode=False,
    pipeline=None)

synthtext_sub_textrecog_train = dict(
    type='OCRDataset',
    data_root=synthtext_textrecog_data_root,
    data_prefix=dict(img_path='synthtext/SynthText_patch_horizontal'),
    ann_file='subset_train_labels.json',
    test_mode=False,
    pipeline=None)
