wildreceipt_data_root = 'data/kie/wildreceipt/'

wildreceipt_train = dict(
    type='WildReceiptDataset',
    data_root=wildreceipt_data_root,
    metainfo=wildreceipt_data_root + 'class_list.txt',
    ann_file='train.txt',
    pipeline=None)

wildreceipt_test = dict(
    type='WildReceiptDataset',
    data_root=wildreceipt_data_root,
    metainfo=wildreceipt_data_root + 'class_list.txt',
    ann_file='test.txt',
    test_mode=True,
    pipeline=None)
