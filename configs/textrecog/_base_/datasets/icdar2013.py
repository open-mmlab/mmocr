icdar2013_textrecog_data_root = 'data/rec/icdar_2013/'

icdar2013_textrecog_train = dict(
    type='OCRDataset',
    data_root=icdar2013_textrecog_data_root,
    ann_file='train_labels.json',
    test_mode=False,
    pipeline=None)

icdar2013_textrecog_test = dict(
    type='OCRDataset',
    data_root=icdar2013_textrecog_data_root,
    ann_file='test_labels.json',
    test_mode=True,
    pipeline=None)
