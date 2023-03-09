svtp_textrecog_data_root = 'data/svtp'

svtp_textrecog_train = dict(
    type='OCRDataset',
    data_root=svtp_textrecog_data_root,
    ann_file='textrecog_train.json',
    pipeline=None)

svtp_textrecog_test = dict(
    type='OCRDataset',
    data_root=svtp_textrecog_data_root,
    ann_file='textrecog_test.json',
    test_mode=True,
    pipeline=None)
