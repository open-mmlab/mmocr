detext_textrecog_data_root = 'data/detext'

detext_textrecog_train = dict(
    type='OCRDataset',
    data_root=detext_textrecog_data_root,
    ann_file='textrecog_train.json',
    pipeline=None)

detext_textrecog_test = dict(
    type='OCRDataset',
    data_root=detext_textrecog_data_root,
    ann_file='textrecog_test.json',
    test_mode=True,
    pipeline=None)
