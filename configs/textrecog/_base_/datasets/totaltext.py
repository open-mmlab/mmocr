ic15_rec_data_root = 'data/totaltext/'

ic15_rec_train = dict(
    type='OCRDataset',
    data_root=ic15_rec_data_root,
    ann_file='textrecog_train.json',
    data_prefix=dict(img_path='crops/'),
    test_mode=False,
    pipeline=None)

ic15_rec_test = dict(
    type='OCRDataset',
    data_root=ic15_rec_data_root,
    ann_file='textrecog_test.json',
    data_prefix=dict(img_path='crops/'),
    test_mode=True,
    pipeline=None)
