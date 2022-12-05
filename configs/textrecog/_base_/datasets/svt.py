svt_textrecog_data_root = 'data/rec/svt/'

svt_textrecog_test = dict(
    type='OCRDataset',
    data_root=svt_textrecog_data_root,
    ann_file='test_labels.json',
    test_mode=True,
    pipeline=None)
