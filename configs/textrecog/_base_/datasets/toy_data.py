toy_data_root = 'tests/data/rec_toy_dataset/'

toy_rec_train = dict(
    type='OCRDataset',
    data_root=toy_data_root,
    data_prefix=dict(img_path='imgs/'),
    ann_file='labels.json',
    pipeline=None,
    test_mode=False)

toy_rec_test = dict(
    type='OCRDataset',
    data_root=toy_data_root,
    data_prefix=dict(img_path='imgs/'),
    ann_file='labels.json',
    pipeline=None,
    test_mode=True)
