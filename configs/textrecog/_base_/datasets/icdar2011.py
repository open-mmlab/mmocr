icdar2011_textrecog_data_root = 'data/rec/icdar_2011/'

icdar2011_textrecog_train = dict(
    type='OCRDataset',
    data_root=icdar2011_textrecog_data_root,
    ann_file='train_labels.json',
    test_mode=False,
    pipeline=None)
