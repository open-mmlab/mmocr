xfund_pt_re_data_root = 'data/xfund/pt'

xfund_pt_re_train = dict(
    type='REDataset',
    data_root=xfund_pt_re_data_root,
    ann_file='re_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

xfund_pt_re_test = dict(
    type='REDataset',
    data_root=xfund_pt_re_data_root,
    ann_file='re_test.json',
    test_mode=True,
    pipeline=None)
