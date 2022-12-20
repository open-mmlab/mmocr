iiit5k_textrecog_data_root = 'data/iiit5k'

iiit5k_textrecog_train = dict(
    type='OCRDataset',
    data_root=iiit5k_textrecog_data_root,
    ann_file='textrecog_train.json',
    pipeline=None)

iiit5k_textrecog_test = dict(
    type='OCRDataset',
    data_root=iiit5k_textrecog_data_root,
    ann_file='textrecog_test.json',
    test_mode=True,
    pipeline=None)
