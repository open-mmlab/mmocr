tt_rec_data_root = 'data/totaltext/'

tt_rec_train = dict(
    type='OCRDataset',
    data_root=tt_rec_data_root,
    ann_file='textrecog_train.json',
    test_mode=False,
    pipeline=None)

tt_rec_test = dict(
    type='OCRDataset',
    data_root=tt_rec_data_root,
    ann_file='textrecog_test.json',
    test_mode=True,
    pipeline=None)
