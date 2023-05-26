thai_manga_data_root = 'tests/data/rec_thai_manga_dataset/'

thai_manga_rec_train = dict(
    type='OCRDataset',
    data_root=thai_manga_data_root,
    data_prefix=dict(img_path='imgs_train/'),
    ann_file='labels_train.json',
    pipeline=None,
    test_mode=False)

thai_manga_rec_test = dict(
    type='OCRDataset',
    data_root=thai_manga_data_root,
    data_prefix=dict(img_path='imgs_test/'),
    ann_file='labels_test.json',
    pipeline=None,
    test_mode=True)
