tt_det_data_root = 'data/totaltext'

tt_det_train = dict(
    type='OCRDataset',
    data_root=tt_det_data_root,
    ann_file='textdet_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

tt_det_test = dict(
    type='OCRDataset',
    data_root=tt_det_data_root,
    ann_file='textdet_test.json',
    test_mode=True,
    pipeline=None)
