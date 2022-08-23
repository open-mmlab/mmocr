iiit5k_rec_data_root = 'data/rec/IIIT5K/'

iiit5k_rec_train = dict(
    type='OCRDataset',
    data_root=iiit5k_rec_data_root,
    ann_file='train_labels.json',
    test_mode=False,
    pipeline=None)

iiit5k_rec_test = dict(
    type='OCRDataset',
    data_root=iiit5k_rec_data_root,
    ann_file='test_labels.json',
    test_mode=True,
    pipeline=None)
