data_root = 'data/det/icdar2017'

train_anno_path = 'instances_training.json'
test_anno_path = 'instances_test.json'

train_dataset = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file=train_anno_path,
    data_prefix=dict(img_path='imgs/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

test_dataset = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file=test_anno_path,
    data_prefix=dict(img_path='imgs/'),
    test_mode=True,
    pipeline=None)

train_list = [train_dataset]
test_list = [test_dataset]
