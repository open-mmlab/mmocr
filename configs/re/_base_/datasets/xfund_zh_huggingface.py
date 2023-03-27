xfund_zh_huggingface_re_data_root = 'data/xfund/zh'

xfund_zh_huggingface_re_train = dict(
    type='REHuggingfaceDataset',
    data_root=xfund_zh_huggingface_re_data_root,
    ann_file='re_train.huggingface',
    pipeline=None)

xfund_zh_huggingface_re_test = dict(
    type='REHuggingfaceDataset',
    data_root=xfund_zh_huggingface_re_data_root,
    ann_file='re_test.huggingface',
    test_mode=True,
    pipeline=None)
