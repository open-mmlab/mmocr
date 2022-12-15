synthtext_add_textrecog_data_root = 'data/rec/synthtext_add/'

synthtext_add_textrecog_train = dict(
    type='OCRDataset',
    data_root=synthtext_add_textrecog_data_root,
    ann_file='train_labels.json',
    test_mode=False,
    pipeline=None)
