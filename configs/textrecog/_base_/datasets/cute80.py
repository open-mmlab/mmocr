cute80_textrecog_data_root = 'data/rec/ct80/'

cute80_textrecog_test = dict(
    type='OCRDataset',
    data_root=cute80_textrecog_data_root,
    ann_file='test_labels.json',
    test_mode=True,
    pipeline=None)
