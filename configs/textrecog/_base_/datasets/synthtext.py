synthtext_textrecog_data_root = 'data/synthtext'

synthtext_textrecog_train = dict(
    type='OCRDataset',
    data_root=synthtext_textrecog_data_root,
    ann_file='textrecog_train.json',
    pipeline=None)

synthtext_sub_textrecog_train = dict(
    type='OCRDataset',
    data_root=synthtext_textrecog_data_root,
    ann_file='subset_textrecog_train.json',
    pipeline=None)

synthtext_an_textrecog_train = dict(
    type='OCRDataset',
    data_root=synthtext_textrecog_data_root,
    ann_file='alphanumeric_textrecog_train.json',
    pipeline=None)
