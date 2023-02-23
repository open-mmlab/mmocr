ctw1500_textspotting_data_root = 'data/CTW1500'

ctw1500_textspotting_train = dict(
    type='AdelDataset',
    data_root=ctw1500_textspotting_data_root,
    ann_file='annotations/train_ctw1500_maxlen25_v2.json',
    data_prefix=dict(img_path='ctwtrain_text_image/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

ctw1500_textspotting_test = dict(
    type='AdelDataset',
    data_root=ctw1500_textspotting_data_root,
    ann_file='annotations/test_ctw1500_maxlen25.json',
    data_prefix=dict(img_path='ctwtest_text_image/'),
    test_mode=True,
    pipeline=None)
