totaltext_textspotting_train = dict(
    type='AdelDataset',
    data_root='spts-data/totaltext',
    ann_file='train.json',
    data_prefix=dict(img_path='train_images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

totaltext_textspotting_test = dict(
    type='OCRDataset',
    data_root='data/totaltext',
    ann_file='textspotting_test.json',
    test_mode=True,
    pipeline=None)
