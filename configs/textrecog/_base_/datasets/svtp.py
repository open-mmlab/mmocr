svtp_rec_data_root = 'data/rec/svtp/'

svtp_rec_test = dict(
    type='OCRDataset',
    data_root=svtp_rec_data_root,
    ann_file='test_labels.json',
    test_mode=True,
    pipeline=None)
