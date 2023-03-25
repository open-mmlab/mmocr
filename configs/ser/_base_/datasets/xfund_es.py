xfund_es_ser_data_root = 'data/xfund/es'

xfund_es_ser_train = dict(
    type='SERDataset',
    data_root=xfund_es_ser_data_root,
    ann_file='ser_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

xfund_es_ser_test = dict(
    type='SERDataset',
    data_root=xfund_es_ser_data_root,
    ann_file='ser_test.json',
    test_mode=True,
    pipeline=None)
