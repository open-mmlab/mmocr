mlt_textspotting_data_root = 'spts-data/mlt2017'

mlt_textspotting_train = dict(
    type='AdelDataset',
    data_root=mlt_textspotting_data_root,
    ann_file='train.json',
    data_prefix=dict(img_path='MLT_train_images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)
