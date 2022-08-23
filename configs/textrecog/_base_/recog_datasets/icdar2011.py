ic11_rec_data_root = 'data/rec/icdar_2011/'

ic11_rec_train = dict(
    type='OCRDataset',
    data_root=ic11_rec_data_root,
    ann_file='train_labels.json',
    test_mode=False,
    pipeline=None)
