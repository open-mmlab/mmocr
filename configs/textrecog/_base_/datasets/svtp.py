svtp_textrecog_data_root = 'data/rec/svtp/'

svtp_textrecog_test = dict(
    type='OCRDataset',
    data_root=svtp_textrecog_data_root,
    ann_file='test_labels.json',
    test_mode=True,
    pipeline=None)
