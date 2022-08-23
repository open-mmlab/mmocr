cute80_rec_data_root = 'data/rec/ct80/'

cute80_rec_test = dict(
    type='OCRDataset',
    data_root=cute80_rec_data_root,
    ann_file='test_labels.json',
    test_mode=True,
    pipeline=None)
