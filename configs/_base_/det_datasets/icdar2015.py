ic15_data_root = 'data/det/icdar2015'

ic15_det_train = dict(
    type='OCRDataset',
    data_root=ic15_data_root,
    ann_file='instances_training.json',
    data_prefix=dict(img_path='imgs/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

ic15_det_test = dict(
    type='OCRDataset',
    data_root=ic15_data_root,
    ann_file='instances_test.json',
    data_prefix=dict(img_path='imgs/'),
    test_mode=True,
    pipeline=None)
