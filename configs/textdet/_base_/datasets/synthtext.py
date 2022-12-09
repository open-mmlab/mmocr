synthtext_textdet_data_root = 'data/synthtext'

synthtext_textdet_train = dict(
    type='OCRDataset',
    data_root=synthtext_textdet_data_root,
    ann_file='textdet_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)
