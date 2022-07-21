st_data_root = 'data/det/synthtext'

st_det_train = dict(
    type='OCRDataset',
    data_root=st_data_root,
    ann_file='instances_training.json',
    data_prefix=dict(img_path='imgs/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

st_det_test = dict(
    type='OCRDataset',
    data_root=st_data_root,
    ann_file='instances_test.json',
    data_prefix=dict(img_path='imgs/'),
    test_mode=True,
    pipeline=None)
