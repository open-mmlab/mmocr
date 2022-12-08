naf_textspotting_data_root = 'data/naf'

naf_textspotting_train = dict(
    type='OCRDataset',
    data_root=naf_textspotting_data_root,
    ann_file='textspotting_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

naf_textspotting_val = dict(
    type='OCRDataset',
    data_root=naf_textspotting_data_root,
    ann_file='textdet_val.json',
    test_mode=True,
    pipeline=None)

naf_textspotting_test = dict(
    type='OCRDataset',
    data_root=naf_textspotting_data_root,
    ann_file='textspotting_test.json',
    test_mode=True,
    pipeline=None)
