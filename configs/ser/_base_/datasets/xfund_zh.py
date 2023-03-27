xfund_zh_ser_data_root = 'data/xfund/zh'

xfund_zh_ser_train = dict(
    type='SERDataset',
    data_root=xfund_zh_ser_data_root,
    ann_file='ser_train.json',
    pipeline=None)

xfund_zh_ser_test = dict(
    type='SERDataset',
    data_root=xfund_zh_ser_data_root,
    ann_file='ser_test.json',
    test_mode=True,
    pipeline=None)
