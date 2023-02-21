naf_textdet_data_root = 'data/naf'

naf_textdet_train = dict(
    type='OCRDataset',
    data_root=naf_textdet_data_root,
    ann_file='textdet_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

naf_textdet_val = dict(
    type='OCRDataset',
    data_root=naf_textdet_data_root,
    ann_file='textdet_val.json',
    test_mode=True,
    pipeline=None)

naf_textdet_test = dict(
    type='OCRDataset',
    data_root=naf_textdet_data_root,
    ann_file='textdet_test.json',
    test_mode=True,
    pipeline=None)
