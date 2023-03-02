icdar2015_textspotting_train = dict(
    type='AdelDataset',
    data_root='spts-data/icdar2015',
    ann_file='ic15_train.json',
    data_prefix=dict(img_path='train_images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

icdar2015_textspotting_test = dict(
    type='OCRDataset',
    data_root='data/icdar2015',
    ann_file='textspotting_test.json',
    test_mode=True,
    pipeline=None)
