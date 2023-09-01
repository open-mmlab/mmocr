detext_textdet_data_root = 'data/detext'

detext_textdet_train = dict(
    type='OCRDataset',
    data_root=detext_textdet_data_root,
    ann_file='textdet_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

detext_textdet_test = dict(
    type='OCRDataset',
    data_root=detext_textdet_data_root,
    ann_file='textdet_test.json',
    test_mode=True,
    pipeline=None)
