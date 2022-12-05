icdar2015_textrecog_data_root = 'data/rec/icdar_2015/'

icdar2015_textrecog_train = dict(
    type='OCRDataset',
    data_root=icdar2015_textrecog_data_root,
    ann_file='train_labels.json',
    test_mode=False,
    pipeline=None)

icdar2015_textrecog_test = dict(
    type='OCRDataset',
    data_root=icdar2015_textrecog_data_root,
    ann_file='test_labels.json',
    test_mode=True,
    pipeline=None)
