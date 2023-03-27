xfund_zh_huggingface_ser_data_root = 'data/xfund/zh'

xfund_zh_huggingface_ser_train = dict(
    type='SERHuggingfaceDataset',
    data_root=xfund_zh_huggingface_ser_data_root,
    ann_file='ser_train.huggingface',
    pipeline=None)

xfund_zh_huggingface_ser_test = dict(
    type='SERHuggingfaceDataset',
    data_root=xfund_zh_huggingface_ser_data_root,
    ann_file='ser_test.huggingface',
    test_mode=True,
    pipeline=None)
