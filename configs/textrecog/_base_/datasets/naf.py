naf_textrecog_data_root = 'data/naf'

naf_textrecog_train = dict(
    type='OCRDataset',
    data_root=naf_textrecog_data_root,
    ann_file='textrecog_train.json',
    pipeline=None)

naf_textrecog_val = dict(
    type='OCRDataset',
    data_root=naf_textrecog_data_root,
    ann_file='textdet_val.json',
    test_mode=True,
    pipeline=None)

naf_textrecog_test = dict(
    type='OCRDataset',
    data_root=naf_textrecog_data_root,
    ann_file='textrecog_test.json',
    test_mode=True,
    pipeline=None)
