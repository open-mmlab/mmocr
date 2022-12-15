icdar2013_textrecog_data_root = 'data/icdar2013'

icdar2013_textrecog_train = dict(
    type='OCRDataset',
    data_root=icdar2013_textrecog_data_root,
    ann_file='textrecog_train.json',
    pipeline=None)

icdar2013_textrecog_test = dict(
    type='OCRDataset',
    data_root=icdar2013_textrecog_data_root,
    ann_file='textrecog_test.json',
    test_mode=True,
    pipeline=None)

icdar2013_857_textrecog_test = dict(
    type='OCRDataset',
    data_root=icdar2013_textrecog_data_root,
    ann_file='textrecog_test_857.json',
    test_mode=True,
    pipeline=None)
