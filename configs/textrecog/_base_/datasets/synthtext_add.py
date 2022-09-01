st_add_rec_data_root = 'data/rec/synthtext_add/'

st_add_rec_train = dict(
    type='OCRDataset',
    data_root=st_add_rec_data_root,
    ann_file='train_labels.json',
    test_mode=False,
    pipeline=None)
