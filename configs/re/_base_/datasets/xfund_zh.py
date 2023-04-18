xfund_zh_re_data_root = 'data/xfund/zh'

xfund_zh_re_train = dict(
    type='XFUNDDataset',
    data_root=xfund_zh_re_data_root,
    ann_file='re_train.json',
    pipeline=None)

xfund_zh_re_test = dict(
    type='XFUNDDataset',
    data_root=xfund_zh_re_data_root,
    ann_file='re_test.json',
    test_mode=True,
    pipeline=None)
