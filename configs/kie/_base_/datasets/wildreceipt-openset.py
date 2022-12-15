wildreceipt_openset_data_root = 'data/wildreceipt/'

wildreceipt_openset_train = dict(
    type='WildReceiptDataset',
    data_root=wildreceipt_openset_data_root,
    metainfo=dict(category=[
        dict(id=0, name='bg'),
        dict(id=1, name='key'),
        dict(id=2, name='value'),
        dict(id=3, name='other')
    ]),
    ann_file='openset_train.txt',
    pipeline=None)

wildreceipt_openset_test = dict(
    type='WildReceiptDataset',
    data_root=wildreceipt_openset_data_root,
    metainfo=dict(category=[
        dict(id=0, name='bg'),
        dict(id=1, name='key'),
        dict(id=2, name='value'),
        dict(id=3, name='other')
    ]),
    ann_file='openset_test.txt',
    test_mode=True,
    pipeline=None)
