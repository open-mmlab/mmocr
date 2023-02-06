syntext2_textspotting_data_root = 'data/syntext2'

syntext2_textspotting_train = dict(
    type='AdelDataset',
    data_root=syntext2_textspotting_data_root,
    ann_file='train.json',
    data_prefix=dict(img_path='emcs_imgs/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)
