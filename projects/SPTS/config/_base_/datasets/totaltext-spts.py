totaltext_textspotting_data_root = 'data/totaltext'

totaltext_textspotting_train = dict(
    type='AdelDataset',
    data_root=totaltext_textspotting_data_root,
    ann_file='train.json',
    data_prefix=dict(img_path='train_images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

totaltext_textspotting_test = dict(
    type='AdelDataset',
    data_root=totaltext_textspotting_data_root,
    ann_file='test.json',
    data_prefix=dict(img_path='test_images/'),
    test_mode=True,
    pipeline=None)
