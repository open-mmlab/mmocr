icdar2013_textspotting_data_root = 'data/icdar2013'

icdar2013_textspotting_train = dict(
    type='AdelDataset',
    data_root=icdar2013_textspotting_data_root,
    ann_file='ic13_train.json',
    data_prefix=dict(img_path='train_images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

icdar2013_textspotting_test = dict(
    type='AdelDataset',
    data_root=icdar2013_textspotting_data_root,
    data_prefix=dict(img_path='test_images/'),
    ann_file='ic13_test.json',
    test_mode=True,
    pipeline=None)
