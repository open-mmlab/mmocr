cute80_textrecog_data_root = 'data/cute80'

cute80_textrecog_test = dict(
    type='OCRDataset',
    data_root=cute80_textrecog_data_root,
    ann_file='textrecog_test.json',
    test_mode=True,
    pipeline=None)
