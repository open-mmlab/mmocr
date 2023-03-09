syntext1_textspotting_data_root = 'spts-data/syntext1'

syntext1_textspotting_train = dict(
    type='AdelDataset',
    data_root=syntext1_textspotting_data_root,
    ann_file='train.json',
    data_prefix=dict(img_path='syntext_word_eng/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)
