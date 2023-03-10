icdar2015_textspotting_data_root = 'spts-data/icdar2015'

icdar2015_textspotting_train = dict(
    type='AdelDataset',
    data_root=icdar2015_textspotting_data_root,
    ann_file='ic15_train.json',
    data_prefix=dict(img_path='train_images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

icdar2015_textspotting_test = dict(
    type='AdelDataset',
    data_root=icdar2015_textspotting_data_root,
    data_prefix=dict(img_path='test_images/'),
    ann_file='ic15_test.json',
    test_mode=True,
    pipeline=None)
